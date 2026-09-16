"""HOOK-4 — a supported test-helper surface: `TEX_Wrangle.tex_testkit`.

The want: `cold_engine_state` and its neighbours should be reachable without an embedding host
loading a file out of a test directory by path. Before this ask the only door in was
`tests/helpers.py`, and TEX's own suite reaches it with a bare `from helpers import *` — which
needs `tests/` on `sys.path`, and putting an embedding host's own `tests/` there risks shadowing
that host's `tests` package.

The lower-risk shape, against adding `tests/__init__.py` (which could move TEX's own pytest
rootdir/collection): a sibling module instead — `TEX_Wrangle/tex_testkit.py` at the package root,
holding the three functions that make up one named set (the "state-isolation kit": `make_img`,
`cold_engine_state`, `armed_profiler`) — and leaves `tests/__init__.py` unwritten.

Four angles: REACHABLE (an embedding host's own interpreter, no `tests/` on `sys.path`),
UNCHANGED (TEX's own `from helpers import *` still yields exactly the base-sha `b7a92e5` name
set — pinned below, not re-derived), NOT-A-FORK (helpers.py's three names ARE tex_testkit's
objects, not copies that can drift apart from underneath both suites), and BOUNDARY (the new
module carries no pytest import and is not reached from the ComfyUI adapter files S-1 already
enumerates).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile

from helpers import *

from TEX_Wrangle import tex_testkit

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # TEX_Wrangle/
_CUSTOM_NODES = os.path.dirname(_PKG)                                        # custom_nodes/
_TESTS_DIR = os.path.join(_PKG, "tests")

_KIT = ("make_img", "cold_engine_state", "armed_profiler")

#: The ComfyUI-facing files S-1 already draws the boundary around
#: (`test_v019_phase2.py::_HOST_FILES`) — the set this ask must never join.
_ADAPTER_FILES = ("tex_node.py", "__init__.py", os.path.join("tex_runtime", "host.py"))

#: `sorted(helpers.__all__)` at v0.35.0 (`b7a92e5`), computed once from that checkout and
#: pinned rather than re-derived from the CURRENT file — an "expected" set re-derived from the
#: same file it is checking can drift in lockstep with a regression and never catch one.
_BASE_SHA = "b7a92e5"
_BASE_ALL = frozenset({
    "sys", "os", "traceback", "math", "re", "shutil", "tempfile", "time", "pickle", "Path",
    "torch",
    "_prepare_output", "_unwrap_latent", "_infer_binding_type", "_map_inferred_type",
    "Lexer", "LexerError", "TokenType",
    "Parser", "ParseError",
    "TypeChecker", "TypeCheckError", "TEXType", "CHANNEL_MAP",
    "TEXMultiError",
    "optimize", "BINDING_HINT_TYPES",
    "Interpreter", "InterpreterError",
    "_ensure_spatial", "_broadcast_pair", "_collect_identifiers",
    "TEXCache",
    "execute_compiled", "_plain_execute", "clear_compiled_cache",
    "try_compile", "_CgBreak", "_CgContinue",
    "TEXStdlib", "SAFE_EPSILON",
    "_perlin2d_fast", "_grad2d_dot", "_lowbias32",
    "SubTestResult", "compile_and_run", "compile_and_infer", "check_code",
    "run_both", "assert_equiv", "check_val", "make_img", "make_latent",
    "make_gradient_frame", "devices",
    "cold_engine_state", "lint_sources", "armed_profiler",
    "_MAX_LOOP_ITERATIONS",
})

#: The child never has `tests/` on `sys.path` — a bare `python -c` starts from the
#: interpreter's own site path plus (on some builds) cwd, and this ask runs the suite from
#: `TEX_Wrangle`'s package parent (never from inside `tests/`) — but the check is made
#: explicit rather than assumed, and `cwd` is pinned below so it cannot depend on whatever
#: directory happened to launch pytest.
_CHILD_SCRIPT = r"""
import sys, os
custom_nodes = {custom_nodes!r}
tests_dir_norm = {tests_dir!r}
already = [p for p in sys.path if p and os.path.normcase(os.path.abspath(p)) == tests_dir_norm]
assert not already, "tests dir already on sys.path: %r" % (already,)
sys.path.insert(0, custom_nodes)
import TEX_Wrangle.tex_testkit as tk
assert set(tk.__all__) == {{"make_img", "cold_engine_state", "armed_profiler"}}, tk.__all__
img = tk.make_img(1, 4, 4, 3, seed=1)
assert tuple(img.shape) == (1, 4, 4, 3), img.shape
with tk.cold_engine_state(warm=False) as cold:
    assert cold.dir and os.path.isdir(cold.dir)
with tk.armed_profiler():
    pass
print("HOOK4-OK")
"""


def _run_child(cache_dir: str):
    script = _CHILD_SCRIPT.format(
        custom_nodes=_CUSTOM_NODES,
        tests_dir=os.path.normcase(os.path.abspath(_TESTS_DIR)),
    )
    env = dict(os.environ, TEX_CACHE_DIR=cache_dir)
    return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          env=env, cwd=_CUSTOM_NODES, timeout=120)


# ── REACHABLE: a second host's own interpreter, no tests/ on sys.path ───────

def test_hook4_tex_testkit_importable_without_tests_on_syspath(r: SubTestResult):
    """The reach HOOK-4 asks for: an embedding host's own interpreter, `TEX_Wrangle`'s package
    parent on `sys.path` and NOTHING under `tests/` — the reach a private path-load under a
    borrowed module name could not give a host without risking that host's own `tests`
    package."""
    print("\n--- HOOK-4: tex_testkit importable with no tests/ on sys.path ---")
    cache_dir = tempfile.mkdtemp(prefix="tex_hook4_child_")
    try:
        proc = _run_child(cache_dir)
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)
    if proc.returncode == 0 and "HOOK4-OK" in proc.stdout:
        r.ok("TEX_Wrangle.tex_testkit imports cleanly from a fresh interpreter")
    else:
        r.fail("HOOK-4 fresh-interpreter import",
               f"rc={proc.returncode}\nstdout={proc.stdout[-800:]}\nstderr={proc.stderr[-800:]}")


# ── UNCHANGED: upstream's bare star-import still yields the base-sha set ────

def test_hook4_bare_star_import_yields_the_base_sha_set(r: SubTestResult):
    """`from helpers import *` must keep yielding exactly the name set it yielded at the base
    sha, whichever of those names now arrive by re-export instead of by local `def`. `import *`
    binds `{n: getattr(mod, n) for n in mod.__all__}` — so this also proves every one of the
    names still resolves as an attribute, not just that the list text is right."""
    print(f"\n--- HOOK-4: helpers.__all__ is unchanged from {_BASE_SHA} ---")
    import helpers as _helpers
    current = set(_helpers.__all__)
    if current == _BASE_ALL:
        r.ok(f"helpers.__all__ is byte-for-byte the {_BASE_SHA} set ({len(current)} names)")
    else:
        missing, extra = _BASE_ALL - current, current - _BASE_ALL
        r.fail("HOOK-4 __all__ drift", f"missing={sorted(missing)} extra={sorted(extra)}")
        return
    unresolved = [n for n in _helpers.__all__ if not hasattr(_helpers, n)]
    if unresolved:
        r.fail("HOOK-4 star-import", f"in __all__ but not an attribute: {unresolved}")
    else:
        r.ok("every name in __all__ resolves -- `from helpers import *` cannot AttributeError")


# ── NOT-A-FORK: helpers.py's three names ARE tex_testkit's objects ──────────

def test_hook4_helpers_reexports_are_tex_testkit_objects(r: SubTestResult):
    """helpers.py's three promoted names must be tex_testkit's own objects — an `is` check,
    not an equality check — or a future edit to one copy silently stops matching the other and
    upstream's suite and a host's suite drift apart from underneath both."""
    print("\n--- HOOK-4: helpers.py re-exports tex_testkit's own objects ---")
    import helpers as _helpers
    if set(tex_testkit.__all__) == set(_KIT):
        r.ok(f"tex_testkit.__all__ is exactly the state-isolation kit: {_KIT}")
    else:
        r.fail("HOOK-4 testkit surface",
               f"got {sorted(tex_testkit.__all__)}, wanted {sorted(_KIT)}")
    drifted = [n for n in _KIT if getattr(_helpers, n) is not getattr(tex_testkit, n)]
    if drifted:
        r.fail("HOOK-4 identity", f"helpers.py holds a COPY, not tex_testkit's object: {drifted}")
    else:
        r.ok("helpers.{make_img,cold_engine_state,armed_profiler} ARE tex_testkit's objects")


# ── BOUNDARY: no pytest import; no adapter file imports tex_testkit ─────────

def test_hook4_testkit_stays_off_pytest_and_the_comfy_adapter_path(r: SubTestResult):
    """The two costs this module must never carry. pytest is a test-runner dependency, never a
    runtime one; the adapter files are S-1's own boundary (`tex_node.py`, `__init__.py`,
    `tex_runtime/host.py`) — a ComfyUI user runs neither suite, and this module reaching them
    would be the first time a test surface did."""
    print("\n--- HOOK-4: tex_testkit imports no pytest; no adapter file imports tex_testkit ---")
    testkit_path = os.path.join(_PKG, "tex_testkit.py")
    try:
        src = open(testkit_path, encoding="utf-8").read()
    except OSError as e:
        r.fail("HOOK-4 boundary", f"can't read {testkit_path}: {e}")
        return
    if re.search(r"(?m)^\s*(?:import\s+pytest\b|from\s+pytest\b)", src):
        r.fail("HOOK-4 boundary", "tex_testkit.py imports pytest")
    else:
        r.ok("tex_testkit.py imports no pytest")

    offenders = []
    for rel in _ADAPTER_FILES:
        path = os.path.join(_PKG, rel)
        try:
            text = open(path, encoding="utf-8").read()
        except OSError:
            continue
        if re.search(r"(?m)^\s*(?:import|from)\s+.*\btex_testkit\b", text):
            offenders.append(rel)
    if offenders:
        r.fail("HOOK-4 boundary", f"adapter file(s) import tex_testkit: {offenders}")
    else:
        r.ok(f"none of {_ADAPTER_FILES} import tex_testkit")
