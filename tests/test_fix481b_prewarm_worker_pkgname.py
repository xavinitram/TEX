"""
FIX-481B -- `prewarm_worker.warm_in_subprocess()` hard-coded the child's import as
`[sys.executable, "-m", "TEX_Wrangle.tex_runtime.prewarm_worker"]`. That token is a bet on
the ComfyUI install's own folder name: it holds under the standing worktree convention
(a `TEX_Wrangle` junction), but nothing guarantees it in the wild. An install whose folder
is the registry's `comfyui-tex-wrangle`, or a plain `TEX` with no junction at all (the MAIN
tree's own real folder name, exactly as named in the bug report) can never start that
child -- the fresh interpreter has no `TEX_Wrangle` on its `sys.path`, so `-m` fails with
`ModuleNotFoundError` and `warm_in_subprocess` reports a non-zero exit, which
`prewarm()`/`prewarm_async()` (before this fix) then dropped on the floor with no trace.

Two things pinned here:
  1. `test_fix481b_child_starts_under_a_renamed_package` -- the actual repro: a real copy of
     the package tree under a DIFFERENT top-level name, imported fresh and driven through
     `warm_in_subprocess()` exactly as `tex_api.prewarm()` does. RED at base (child fails to
     import, `result["error"]` set); GREEN at head (the boundary derives the real import
     name from its own `__package__` instead of assuming `TEX_Wrangle`).
  2. `test_fix481b_subprocess_failure_falls_back_to_thread_and_surfaces_error` -- the other
     half of the same ask: even when the child DOES fail (this row simulates it
     deterministically rather than re-breaking the import), the deferred jobs must still get
     warmed (fallback to the original "thread" mechanism) and the caller must be able to see
     that it happened (`summary["subprocess_error"]`). RED at base (no such key exists at
     all, and the deferred jobs are simply never warmed); GREEN at head.

No CUDA needed for either row: (1) drives `warm_in_subprocess()` directly on `device="cpu"`,
which needs no compile backend at all; (2) reproduces the CUDA-only code path by
monkeypatching the same three gates `tests/test_prewarm481_gil_bound.py` already does
(`torch.cuda.is_available`, `compiled._cuda_headroom_ok`, `compiled._capture_in_flight`),
plus `compiled._try_compile` itself, so it runs identically on a CPU-only box.
"""
from __future__ import annotations

import importlib
import os
import shutil
import sys

import pytest

from helpers import *  # noqa: F401,F403  (TEXType, cold_engine_state)
from TEX_Wrangle import tex_api

# ── shared fixtures ──────────────────────────────────────────────────────────

# Top-level entries that are not needed to IMPORT the package (tests/tools/benchmarks/docs
# are never imported by `tex_api`'s own chain; `.git`/`.github`/`__pycache__` must never be
# copied either way) -- skipping them keeps the copy fast and small instead of duplicating
# the whole ~40 MB tree for a test that only needs the importable Python package.
_SKIP_TOP = {"tests", "tools", "benchmarks", "docs", "js", "editor_build", "assets",
             "stock", ".git", ".github", "__pycache__"}


def _copy_package_tree(src: str, dst: str) -> None:
    def _ignore(dirpath, names):
        ignored = {"__pycache__"}
        if os.path.abspath(dirpath) == os.path.abspath(src):
            ignored |= (_SKIP_TOP & set(names))
        return ignored
    shutil.copytree(src, dst, ignore=_ignore)


def _two_programs():
    out = []
    for i in range(2):
        code = f"vec3 c=@A.rgb*1.2 - 0.{i}; c=clamp(c,0.0,1.0); @OUT=vec4(c,1.0);"
        out.append((code, {"A": TEXType.VEC3}))
    return out


# ── 1. the actual repro: a renamed package, driven through the real boundary ───────────

def test_fix481b_child_starts_under_a_renamed_package(tmp_path, r: SubTestResult):
    print("\n--- FIX-481B: warm_in_subprocess()'s child must start under ANY package "
          "name, not just TEX_Wrangle ---")
    import TEX_Wrangle
    src_root = os.path.dirname(os.path.abspath(TEX_Wrangle.__file__))
    # "TEX" -- deliberately the SAME rename the bug report names ("a plain TEX folder with
    # no junction"), and, not coincidentally, the main tree's own real folder name.
    dst_root = os.path.join(str(tmp_path), "TEX")
    _copy_package_tree(src_root, dst_root)
    container = str(tmp_path)

    sys.path.insert(0, container)
    try:
        mod = importlib.import_module("TEX.tex_runtime.prewarm_worker")
        source = "vec3 c=@A.rgb*1.3 - 0.1; c=clamp(c,0.0,1.0); @OUT=vec4(c,1.0);"
        jobs = [(source, {"A": TEXType.VEC3.value}, "fix481b-fake-fp")]
        result = mod.warm_in_subprocess(jobs, device="cpu", precision="fp32",
                                        compile_mode="none")
    finally:
        try:
            sys.path.remove(container)
        except ValueError:
            pass
        for name in list(sys.modules):
            if name == "TEX" or name.startswith("TEX."):
                del sys.modules[name]

    try:
        assert result.get("error") is None, (
            f"child failed to start/import under a renamed package ({dst_root!r} on "
            f"sys.path as 'TEX'): {result}")
        assert result.get("programs") == 1 and result.get("errors", 0) == 0, result
        r.ok(f"child started + warmed under a renamed package: {result}")
    except AssertionError as e:
        r.fail("FIX-481B renamed-package child start", str(e))


# ── 2. the other half: a genuine child failure falls back, and is surfaced ─────────────

def test_fix481b_subprocess_failure_falls_back_to_thread_and_surfaces_error(r: SubTestResult):
    print("\n--- FIX-481B: a failed warm subprocess must fall back to the in-process warm "
          "and surface the failure, never silently warm nothing ---")
    with cold_engine_state():
        from TEX_Wrangle.tex_runtime import compiled as C
        from TEX_Wrangle.tex_runtime import prewarm_worker as _pw_sub
        import torch as _torch

        real_is_available = _torch.cuda.is_available
        real_headroom = C._cuda_headroom_ok
        real_capture = C._capture_in_flight
        real_try_compile = C._try_compile
        real_warm_subprocess = _pw_sub.warm_in_subprocess

        _torch.cuda.is_available = lambda: True
        C._cuda_headroom_ok = lambda *a, **kw: True
        C._capture_in_flight = lambda: False
        C._try_compile = lambda *a, **kw: object()   # fast, deterministic "compiled" stand-in

        def _boom(jobs, **kw):
            return {"bg_compile": 0,
                     "error": "worker exited 1: ModuleNotFoundError (simulated FIX-481B repro)"}
        _pw_sub.warm_in_subprocess = _boom

        try:
            programs = _two_programs()
            summary = tex_api.prewarm(programs, device="cuda", precision="fp32",
                                      compile_mode="auto", bg_compile_mode="subprocess")
            # Drain whatever the fallback submitted so this test's own assertions never race
            # a background thread still warming when it reads `summary` (the summary dict
            # itself is already final by the time `prewarm()` returns -- this only avoids
            # leaving a dangling future for a LATER test to trip over).
            for _ck, _fut in list(C._bg_futures.items()):
                try:
                    _fut.result(timeout=10)
                except Exception:
                    pass
        finally:
            _torch.cuda.is_available = real_is_available
            C._cuda_headroom_ok = real_headroom
            C._capture_in_flight = real_capture
            C._try_compile = real_try_compile
            _pw_sub.warm_in_subprocess = real_warm_subprocess

        try:
            assert summary.get("subprocess_error"), (
                f"a failed warm subprocess must surface on the summary "
                f"(summary['subprocess_error']), got: {summary}")
            assert summary["programs"] == 2 and summary["errors"] == 0, summary
            assert summary["bg_compile"] == 2, (
                f"both deferred jobs should have fallen back to the in-process 'thread' "
                f"warm instead of being dropped: {summary}")
            r.ok(f"subprocess failure surfaced AND both jobs fell back: {summary}")
        except AssertionError as e:
            r.fail("FIX-481B subprocess-failure fallback", str(e))
