"""FIX-HYGIENE H3 (v0.50.0 Phase C, B4#3) — `tools/gen_function_reference.py` must not wipe
`Function-Reference.md` when run the way its own module docstring tells a human to run it:
`python tools/gen_function_reference.py`, standalone, in a FRESH process.

Root cause (confirmed by the audit that filed this): `stdlib_registry.REGISTRY` is only
populated as a SIDE EFFECT of importing `tex_runtime.stdlib` (the `@stdlib(...)` decorators
sit on the `fn_*` impls in `stdlib_core`/`stdlib_sample`/etc., which `tex_runtime.stdlib`
pulls in at its own top level) — but the generator imports `stdlib_registry` directly and
never anything that imports `tex_runtime.stdlib`. Under pytest, `stdlib` is already imported
by the time any test runs (something in collection imports it first), so `R.REGISTRY` is
never empty in-suite and the drift check (`test_doc4_reference`) stays green — the bug is
invisible from inside the suite and only bites a human (or a script) running the generator
on its own, in a process that has imported nothing else yet.

This runs the generator as a REAL standalone subprocess (`sys.executable
tools/gen_function_reference.py`, no pytest, no prior import of anything) against an
isolated COPY of the tree, never the real one — the audit that found this bug mutated the
real, committed `Function-Reference.md` by running the script directly against the live
checkout and had to `git checkout --` to undo it; this test exists to prove the fix without
repeating that mistake. Red at base: the copy's `Function-Reference.md` comes back an
(empty-bodied) stub reporting "0/0 functions documented". Green at head: the regenerated
file matches the tree's own committed reference exactly (mirroring `test_doc4_reference`'s
own in-process comparison, here reproduced across a real process boundary).
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from helpers import SubTestResult

_PKG = Path(__file__).resolve().parent.parent

#: Same exclusion set `test_house50_h4_junctioned_collection.py` uses for its own
#: scratch-copy scan surface — `.git` (this copy must not carry one; a subprocess run here
#: never needs history) and `__pycache__` (speed/determinism only).
_EXCLUDE_DIRS = frozenset({".git", "__pycache__"})


def _copy_tree(dst: Path) -> None:
    def _ignore(dirpath, names):
        return [n for n in names if n in _EXCLUDE_DIRS]
    shutil.copytree(_PKG, dst, ignore=_ignore, dirs_exist_ok=True)


def test_fixhygiene_h3_standalone_generator_does_not_wipe_the_reference(r: SubTestResult):
    print("\n--- FIX-HYGIENE H3: tools/gen_function_reference.py run standalone must not "
          "wipe Function-Reference.md ---")
    committed = (_PKG / "Function-Reference.md").read_text(encoding="utf-8").replace("\r\n", "\n")

    with tempfile.TemporaryDirectory(prefix="fixhygiene_h3_") as tmp:
        # The generator's own path arithmetic (`ROOT = dirname(HERE)`) plus its
        # `from TEX_Wrangle.tex_runtime import ...` import both require the copy's own
        # top directory to be named exactly this -- not a junction/symlink test (that
        # class is covered elsewhere), just the plain import-name requirement.
        copy_root = Path(tmp) / "TEX_Wrangle"
        try:
            _copy_tree(copy_root)
        except OSError as e:
            r.fail("H3 scratch copy", f"could not build the isolated copy: {e}")
            return

        try:
            proc = subprocess.run(
                [sys.executable, "tools/gen_function_reference.py"],
                cwd=str(copy_root), capture_output=True, text=True, timeout=60,
            )
        except subprocess.TimeoutExpired:
            r.fail("H3 standalone run", "the generator did not finish inside 60s")
            return

        out = proc.stdout + proc.stderr
        if proc.returncode != 0:
            r.fail("H3 standalone run", f"rc={proc.returncode}: {out[-1500:]}")
            return
        if "0/0 functions documented" in out:
            r.fail("H3 standalone run",
                   f"the generator reported 0/0 functions documented -- the registry was "
                   f"empty, i.e. the wipe bug is back: {out.strip()!r}")
            return

        regen = (copy_root / "Function-Reference.md").read_text(encoding="utf-8").replace("\r\n", "\n")
        if regen != committed:
            # A short, actionable diff rather than the whole (multi-hundred-line) file.
            c_lines, r_lines = committed.splitlines(), regen.splitlines()
            first_diff = next((i for i in range(min(len(c_lines), len(r_lines)))
                               if c_lines[i] != r_lines[i]), min(len(c_lines), len(r_lines)))
            r.fail("H3 standalone run",
                   f"standalone regen (len {len(regen)}) != committed reference "
                   f"(len {len(committed)}); first differing line {first_diff}: "
                   f"committed={c_lines[first_diff:first_diff + 1]!r} "
                   f"regen={r_lines[first_diff:first_diff + 1]!r}")
            return
        r.ok(f"standalone run reproduced the committed Function-Reference.md exactly "
             f"({out.strip().splitlines()[-1] if out.strip() else ''!r})")
