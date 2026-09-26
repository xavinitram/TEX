"""FIX-OBSROUTE (v0.46 Phase C) — R1 [B4#1 HIGH]: import-order independence.

`tex_engine_tiers.py` used to bind `_tex_engine` EAGERLY (`from . import tex_engine as
_tex_engine` at module scope). That is safe when `tex_engine` is imported FIRST (it is
already mid-import, sitting in `sys.modules`, by the time `tex_engine_tiers` is reached
from inside `tex_engine.py`'s own SPLIT-E re-export). It is NOT safe the other way round:
importing `tex_engine_tiers` FIRST, in a fresh process, starts `tex_engine.py`'s own body
running (nothing has put it in `sys.modules` yet), which reaches its own
`from .tex_engine_tiers import (select_tier, ...)` re-export while `tex_engine_tiers`'s
body is still stuck on the very import line that started this chain — `tex_engine_tiers`
is in `sys.modules` by then, but none of its functions exist yet, so that re-export raises
`ImportError: cannot import name 'select_tier' from partially initialized module`.

Fixed by resolving `tex_engine_tiers`'s `_tex_engine` reference LAZILY (a proxy whose
`__getattr__` imports `tex_engine` on first attribute access, long after both modules have
finished loading either way) instead of eagerly at import time.

**Follow-up.** SPLIT-I's three interpreter mixin modules (`interpreter_binding.py` /
`interpreter_control_flow.py` / `interpreter_spatial.py`) turned out to share the exact
same defect class: each carried a top-level `from .interpreter import NAME, ...` for
names `interpreter.py` imports THEM before defining (the mixin classes are composed by
inheritance at `interpreter.py`'s own class-definition time), so importing any one of the
three first, in a fresh process, ImportErrored the identical way `tex_engine_tiers` did.
Each of those three modules' own docstring already claimed the names it does not own are
"imported back lazily (inside the methods that need them)" — true for MOST of the names in
each file, but not the ones that happened to be defined early enough in `interpreter.py`'s
own body to work when `interpreter` is imported first. Fixed the same way those already-
lazy names were done: moved into the methods that use them, no lazy-proxy class needed
(unlike `tex_engine_tiers`, nothing here binds a whole module reference — each mixin only
ever needs a handful of names, at call time, never at class-definition time). Checked
`tex_results_residency.py` (SPLIT-R) and `tex_results_keys.py` (NEG-6), the other split-out
mixin/leaf siblings in the tree, and the STR-7 codegen split's own siblings
(`codegen_masked.py`, `codegen_stdfns.py`) — none of them import back from their "parent"
module at module scope at all, so none of them share this defect.

This test parametrizes over EVERY product module — every top-level `TEX_Wrangle/*.py` and
every `TEX_Wrangle/tex_runtime/*.py` — imported ALONE, first, in its own fresh subprocess,
with NO exclusions. The list is discovered from the actual file tree (not hand-enumerated)
so it never goes stale as modules are added or split."""
import glob as _glob
import pathlib
import subprocess
import sys as _sys

from helpers import SubTestResult   # noqa: F401  (imported for type/documentation parity)


def _custom_nodes_dir() -> str:
    import TEX_Wrangle
    return str(pathlib.Path(TEX_Wrangle.__file__).resolve().parent.parent)


def _discover_modules() -> list:
    """Every product module dotted name, `TEX_Wrangle.X` and `TEX_Wrangle.tex_runtime.X`,
    for every `*.py` file that is not a test, not `__init__.py`, and not a private/ignored
    helper. Excludes `tex_runtime/__init__.py` itself (that one is exercised by importing
    the PACKAGE, not a leaf) and this suite's own `tests/` tree. NO other exclusions — every
    product module must import first, alone, in a fresh process."""
    pkg_root = pathlib.Path(__file__).resolve().parent.parent  # .../TEX_Wrangle
    names = []
    for f in sorted(_glob.glob(str(pkg_root / "*.py"))):
        stem = pathlib.Path(f).stem
        if stem in ("__init__",) or stem.startswith("test_"):
            continue
        names.append(f"TEX_Wrangle.{stem}")
    for f in sorted(_glob.glob(str(pkg_root / "tex_runtime" / "*.py"))):
        stem = pathlib.Path(f).stem
        if stem in ("__init__",) or stem.startswith("test_"):
            continue
        names.append(f"TEX_Wrangle.tex_runtime.{stem}")
    return names


def _import_first_in_fresh_process(dotted: str, custom_nodes: str) -> tuple:
    """Returns (ok, message). A fresh interpreter, `dotted` is the FIRST and ONLY thing
    touched off the package."""
    code = (
        "import sys\n"
        f"sys.path.insert(0, {custom_nodes!r})\n"
        f"import {dotted}\n"
        "print('IMPORT_OK')\n"
    )
    try:
        proc = subprocess.run([_sys.executable, "-X", "utf8", "-c", code],
                              capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return False, "subprocess timed out after 60s"
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()
        return False, tail[-1] if tail else f"exit code {proc.returncode}"
    return "IMPORT_OK" in proc.stdout, (proc.stdout or "").strip()


def test_r1_every_product_module_imports_first_in_a_fresh_process(r: SubTestResult):
    print("\n--- FIX-OBSROUTE R1: every product module imports first, alone, cleanly ---")
    custom_nodes = _custom_nodes_dir()
    modules = _discover_modules()
    if len(modules) < 40:
        r.fail("FIX-OBSROUTE R1 discovery",
               f"only found {len(modules)} candidate modules — the glob-based discovery "
               f"probably broke (expected 60+); refusing to report a false-green pass")
        return

    bad = []
    tiers_row_seen = False
    for dotted in modules:
        if dotted == "TEX_Wrangle.tex_engine_tiers":
            tiers_row_seen = True
        ok, msg = _import_first_in_fresh_process(dotted, custom_nodes)
        if ok:
            r.ok(f"{dotted}: imports first in a fresh process")
        else:
            bad.append(f"{dotted}: {msg}")

    if not tiers_row_seen:
        r.fail("FIX-OBSROUTE R1 discovery", "tex_engine_tiers was not in the discovered "
               "module list — the one row this ask exists to prove would be silently "
               "skipped")
    if bad:
        r.fail("FIX-OBSROUTE R1", "; ".join(bad))
    else:
        r.ok(f"all {len(modules)} product modules import first, alone, in their own fresh "
             f"process (including tex_engine_tiers, the row R1 exists for)")
