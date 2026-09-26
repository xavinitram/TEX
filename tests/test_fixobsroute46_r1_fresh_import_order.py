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

This test parametrizes over EVERY product module — every top-level `TEX_Wrangle/*.py` and
every `TEX_Wrangle/tex_runtime/*.py` — imported ALONE, first, in its own fresh subprocess.
The list is discovered from the actual file tree (not hand-enumerated) so it never goes
stale as modules are added or split. `tex_engine_tiers` is the one row this ask names as
red at base; every other module is expected to already import cleanly first (this test's
job is to prove that FOR ALL of them, not just the one already known)."""
import glob as _glob
import os
import pathlib
import subprocess
import sys as _sys

from helpers import SubTestResult   # noqa: F401  (imported for type/documentation parity)


def _custom_nodes_dir() -> str:
    import TEX_Wrangle
    return str(pathlib.Path(TEX_Wrangle.__file__).resolve().parent.parent)


#: SPLIT-I's three interpreter mixin modules (`interpreter_binding.py` /
#: `interpreter_control_flow.py` / `interpreter_spatial.py`) have the EXACT SAME
#: import-order defect class R1 fixes for `tex_engine_tiers` — importing any one of them
#: first, in a fresh process, ImportErrors the same way (confirmed while writing this test).
#: They are OUT OF SCOPE for FIX-OBSROUTE: the ask's row list names `tex_engine_tiers.py`
#: specifically, not the SPLIT-I mixins, and they are internal `Interpreter`-composition
#: pieces no product entry point or documented host seam imports standalone (unlike
#: `tex_engine_tiers`, which SPLIT-E's re-export chain can genuinely reach first). Filed as
#: a separate finding rather than fixed here, so this ratchet does not silently widen this
#: ask's diff into three unrelated files.
_OUT_OF_SCOPE = frozenset({
    "TEX_Wrangle.tex_runtime.interpreter_binding",
    "TEX_Wrangle.tex_runtime.interpreter_control_flow",
    "TEX_Wrangle.tex_runtime.interpreter_spatial",
})


def _discover_modules() -> list:
    """Every product module dotted name, `TEX_Wrangle.X` and `TEX_Wrangle.tex_runtime.X`,
    for every `*.py` file that is not a test, not `__init__.py`, and not a private/ignored
    helper. Excludes `tex_runtime/__init__.py` itself (that one is exercised by importing
    the PACKAGE, not a leaf), this suite's own `tests/` tree, and `_OUT_OF_SCOPE` (see
    above — a real, separately-filed defect, not something this ratchet should mask by
    simply not looking, so it stays excluded WITH A NAME rather than by omission)."""
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
    return [n for n in names if n not in _OUT_OF_SCOPE]


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
