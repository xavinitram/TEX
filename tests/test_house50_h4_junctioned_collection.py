"""HOUSE-50/H4 (TRK-227) — the suite's own collection must not silently ModuleNotFoundError
on a checkout shaped the way TRK-227 named: no `.git` at all, under a directory NOT
literally called `TEX_Wrangle`, reached only through a `TEX_Wrangle` junction/symlink of
that name beside it -- the exact convention the ComfyUI install itself uses
(`custom_nodes/TEX` + a `custom_nodes/TEX_Wrangle` junction) and the one `GPU-49`'s own
bench trees use.

TRK-227's own four confirmed rows (`test_fixgate47_g5_commit_message_scan.py::
test_fixgate_g5_cli_exit_codes`, `test_gateverdict_infra_red.py::
test_gateverdict_refuses_when_not_importable_as_tex_wrangle`, `test_v019_phase2.py::
test_s4_validate_hw_console_cp1252_safe`, `test_v046_fixgate.py::
test_g5_prune_is_wired_into_a_gate_run`) were never reproduced standalone against a tree
shaped exactly this way (checked here, individually, against a real junctioned/no-`.git`
copy: all four pass) -- the tracker's own row already says this is a full-suite-scale
property, not a per-test defect. What IS cheaply, deterministically checkable without a
~2000-test run is the more basic claim underneath all of them: that COLLECTING the whole
suite from this exact shape does not itself ModuleNotFoundError, which is the failure mode
`gate.py::_importable_as_tex_wrangle` exists to refuse EARLY rather than let happen as a
wall of tracebacks (`tests/test_gateverdict_infra_red.py`'s own canary covers `gate.py`'s
refusal; this file covers plain `pytest --collect-only`, which never asks `gate.py` at
all -- a human or CI runner can reach this shape directly).
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from helpers import SubTestResult
from test_simp5_citations import _make_link   # reuse NEG-4's own portable link helper

_PKG = Path(__file__).resolve().parent.parent

#: What gets left OUT of the scratch copy. Started narrow (just the test/runtime dirs)
#: and widened after this test's own first run caught the gap: `test_perf7_compiled_cold.py`
#: loads `benchmarks/host_path_counts.py` at COLLECTION time (`helpers.load_counts_harness`),
#: so a "collection-relevant surface" guess that excludes `benchmarks/` is exactly the kind
#: of narrow copy that would have made this check pass for the wrong reason. `.git` is
#: excluded on purpose (this shape has none); `__pycache__` for speed/determinism only.
_EXCLUDE_DIRS = frozenset({".git", "__pycache__"})


def _copy_collection_surface(dst: Path) -> None:
    def _ignore(dirpath, names):
        return [n for n in names if n in _EXCLUDE_DIRS]
    shutil.copytree(_PKG, dst, ignore=_ignore, dirs_exist_ok=True)


def test_house50_h4_collection_from_a_junctioned_no_git_checkout(r: SubTestResult):
    print("\n--- HOUSE-50/H4: whole-suite collection survives a junctioned, no-.git, "
          "differently-named checkout ---")
    with tempfile.TemporaryDirectory(prefix="house50_h4_") as tmp:
        tmp_path = Path(tmp)
        real = tmp_path / "TEX_extracted_not_wrangle"
        link = tmp_path / "TEX_Wrangle"
        try:
            _copy_collection_surface(real)
        except OSError as e:
            r.fail("H4 scratch copy", f"could not build the collection-surface copy: {e}")
            return
        if (real / ".git").exists():
            r.fail("H4 scratch copy", "the scratch copy must not carry a .git (it did)")
            return

        reason = _make_link(link, real)
        if reason is not None:
            r.skip("H4 junctioned collection reproduction",
                   f"this platform could create neither a junction nor a symlink: {reason}")
            return

        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "TEX_Wrangle/tests",
                 "--collect-only", "-q", "-p", "no:cacheprovider"],
                cwd=str(tmp_path), capture_output=True, text=True, timeout=120,
            )
        except subprocess.TimeoutExpired:
            r.fail("H4 junctioned collection", "collection did not finish inside 120s")
            return
        finally:
            # Junction/symlink first (a plain directory-entry removal, not recursive into
            # the target) -- removing the real copy first would leave a dangling link this
            # temp dir's own cleanup then has to fight with on some platforms.
            try:
                if sys.platform == "win32":
                    os.rmdir(str(link))
                else:
                    os.unlink(str(link))
            except OSError:
                pass

        out = proc.stdout + proc.stderr
        if "ModuleNotFoundError" in out:
            r.fail("H4 junctioned collection",
                   f"collection hit ModuleNotFoundError from this checkout shape "
                   f"(rc={proc.returncode}): {out[-1500:]}")
        elif proc.returncode not in (0, 5):   # 5 = pytest's own "no tests collected"
            r.fail("H4 junctioned collection",
                   f"unexpected collection rc={proc.returncode}: {out[-1500:]}")
        else:
            n = sum(1 for ln in out.splitlines() if "::" in ln)
            r.ok(f"the whole suite collects cleanly from a junctioned, no-.git, "
                 f"differently-named checkout (rc={proc.returncode}, ~{n} item line(s))")
