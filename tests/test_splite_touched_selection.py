"""SPLIT-E — `tools/gate.py --tier touched`'s selection logic.

The ask's rule ("test files that import a touched module, plus ALWAYS the docs/citation/
mutation/seam/skip-budget/floors ratchets, always") is a claim about CODE, not prose, so
this pins it directly against `tools/gate.py`'s own functions — `_touched_module` (path ->
dotted module name), `_test_module_refs` (a test file's own imports -> the dotted module
names it could resolve to) and `select_touched_tests` (the union: name-matches + import-
matches + the fixed ALWAYS set) — rather than trusting the tier's printed line, the same
posture `tests/test_gateverdict_infra_red.py` already takes toward `_run`'s dead-leg guard.

Loads `tools/gate.py` by path (it is not a package — `.comfyignore`d like `tests/` and
`benchmarks/`), mirroring `test_gateverdict_infra_red.py::_gate` and
`test_simp1_gate.py`'s own loader so this tests the tool the law tells an implementer to
run, not a copy of its logic.

PORTABILITY: pure stdlib (`tempfile`, `ast` via the module under test) plus a monkeypatched
`_git`/`_PKG`, so this needs no torch, no CUDA, no ComfyUI, no real git history and no real
`origin/main` — a worktree that has never fetched still runs this file identically.
"""
import os
import sys
import tempfile

from helpers import SubTestResult


def _gate():
    """Load `tools/gate.py` by path, once per process (mirrors
    `test_gateverdict_infra_red.py::_gate` / `test_simp1_gate.py`)."""
    mod = sys.modules.get("_splite_gate")
    if mod is not None:
        return mod
    import importlib.util
    import pathlib
    path = pathlib.Path(__file__).resolve().parent.parent / "tools" / "gate.py"
    spec = importlib.util.spec_from_file_location("_splite_gate", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_splite_gate"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_splite_touched_module_path_mapping(r: SubTestResult):
    print("\n--- SPLIT-E: _touched_module maps a touched path to its dotted module name ---")
    g = _gate()
    cases = [
        ("tex_engine.py", "tex_engine"),
        ("tex_engine_tiers.py", "tex_engine_tiers"),
        ("tex_runtime/compiled.py", "tex_runtime.compiled"),
        ("tex_runtime/interpreter_spatial.py", "tex_runtime.interpreter_spatial"),
        ("tex_compiler/ast_nodes.py", "tex_compiler.ast_nodes"),
        ("tex_io/exr.py", "tex_io.exr"),
        # Not importable product modules -- caught by the ALWAYS set, never by import-matching.
        ("tests/test_v017_phase2.py", None),
        ("tools/gate.py", None),
        ("benchmarks/bench_engine.py", None),
        ("docs/roadmap.md", None),
        ("AGENTS.md", None),
        ("__init__.py", None),
        ("editor_build/tex.js", None),
    ]
    try:
        bad = [(p, want, g._touched_module(p)) for p, want in cases
              if g._touched_module(p) != want]
        assert not bad, f"mismatch(es): {bad}"
        r.ok(f"all {len(cases)} path -> module case(s) match")
    except Exception as e:
        r.fail("SPLIT-E _touched_module", f"{type(e).__name__}: {e}")


def test_splite_test_module_refs_both_import_shapes(r: SubTestResult):
    print("\n--- SPLIT-E: _test_module_refs reads both import shapes a test file uses ---")
    g = _gate()
    src = (
        "from TEX_Wrangle import tex_engine, tex_chain\n"
        "from TEX_Wrangle.tex_runtime import tier_trace\n"
        "from TEX_Wrangle.tex_runtime.compiled import execute_compiled\n"
        "import TEX_Wrangle.tex_memory\n"
        "import os\n"          # a plain stdlib import must NOT show up
        "from helpers import *\n"   # a non-TEX_Wrangle from-import must NOT show up
    )
    want = {"tex_engine", "tex_chain", "tex_runtime", "tex_runtime.compiled", "tex_memory"}
    try:
        with tempfile.TemporaryDirectory(prefix="tex-splite-") as d:
            path = os.path.join(d, "test_sample.py")
            with open(path, "w", encoding="utf-8") as f:
                f.write(src)
            refs = g._test_module_refs(path)
        assert refs == want, f"got {sorted(refs)}, want {sorted(want)}"
        r.ok(f"parsed refs {sorted(refs)} exactly")
    except Exception as e:
        r.fail("SPLIT-E _test_module_refs", f"{type(e).__name__}: {e}")


def test_splite_test_module_refs_unparseable_file_is_empty_not_raising(r: SubTestResult):
    print("\n--- SPLIT-E: _test_module_refs never raises, even on unparseable input ---")
    g = _gate()
    try:
        with tempfile.TemporaryDirectory(prefix="tex-splite-") as d:
            path = os.path.join(d, "test_broken.py")
            with open(path, "w", encoding="utf-8") as f:
                f.write("this is not ) valid python (\n")
            refs = g._test_module_refs(path)
        assert refs == set(), f"expected the empty set on a parse failure, got {refs!r}"
        r.ok("an unparseable file yields the empty set, not an exception")
    except Exception as e:
        r.fail("SPLIT-E _test_module_refs (broken file)", f"{type(e).__name__}: {e}")


def test_splite_select_touched_tests_unions_name_import_and_always(r: SubTestResult):
    print("\n--- SPLIT-E: select_touched_tests unions name-matches + import-matches + ALWAYS ---")
    g = _gate()
    try:
        with tempfile.TemporaryDirectory(prefix="tex-splite-pkg-") as pkg:
            tests_dir = os.path.join(pkg, "tests")
            os.makedirs(tests_dir)
            # (a) imports the touched module -- must be selected by import-matching.
            with open(os.path.join(tests_dir, "test_imports_touched.py"), "w",
                     encoding="utf-8") as f:
                f.write("from TEX_Wrangle import tex_engine\n")
            # (b) imports something else entirely -- must NOT be selected.
            with open(os.path.join(tests_dir, "test_unrelated.py"), "w", encoding="utf-8") as f:
                f.write("from TEX_Wrangle import tex_lazy\n")
            # (c) a non-`test_*.py` file that WOULD match by import -- must be ignored: the
            #     scan only walks `test_*.py`, exactly like pytest's own collection would.
            with open(os.path.join(tests_dir, "helpers.py"), "w", encoding="utf-8") as f:
                f.write("from TEX_Wrangle import tex_engine\n")

            orig_pkg, orig_git = g._PKG, g._git

            def _fake_git(*args, cwd=None):
                if args[:2] == ("rev-parse", "--verify"):
                    return "deadbeef\n"
                if args and args[0] == "diff":
                    # The diff touched a product module (import-match) AND, separately, a
                    # test file directly by name (name-match) -- both selection rules fire.
                    return "tex_engine.py\ntests/test_imports_touched.py\n"
                return ""

            g._PKG, g._git = pkg, _fake_git
            try:
                files, touched_mods, base_resolved = g.select_touched_tests("origin/main")
            finally:
                g._PKG, g._git = orig_pkg, orig_git

            always = {p for _, p in g._ALWAYS_TOUCHED}
            want = always | {"tests/test_imports_touched.py"}
            assert base_resolved, "a fake git that answers rev-parse must resolve the base"
            assert "tex_engine" in touched_mods, f"touched_mods={touched_mods}"
            assert set(files) == want, (
                f"got {sorted(files)}, want {sorted(want)} -- test_unrelated.py must be "
                f"excluded (imports a module that was not touched) and helpers.py must be "
                f"excluded (not a test_*.py file) regardless of what it imports")
        r.ok(f"selected exactly the ALWAYS set plus the name- and import-matched file(s): "
             f"{sorted(files)}")
    except Exception as e:
        r.fail("SPLIT-E select_touched_tests union", f"{type(e).__name__}: {e}")


def test_splite_select_touched_tests_unresolved_base_degrades_to_always(r: SubTestResult):
    print("\n--- SPLIT-E: an unresolved --base degrades to ALWAYS-only and says so ---")
    g = _gate()
    try:
        with tempfile.TemporaryDirectory(prefix="tex-splite-pkg2-") as pkg:
            os.makedirs(os.path.join(pkg, "tests"))
            orig_pkg, orig_git = g._PKG, g._git
            g._PKG, g._git = pkg, (lambda *a, cwd=None: "")   # every git call "fails"
            try:
                files, touched_mods, base_resolved = g.select_touched_tests("no-such-ref")
            finally:
                g._PKG, g._git = orig_pkg, orig_git

            always = {p for _, p in g._ALWAYS_TOUCHED}
            assert base_resolved is False, "an empty rev-parse answer must not resolve"
            assert touched_mods == set(), f"expected no touched modules, got {touched_mods}"
            assert set(files) == always, (
                f"got {sorted(files)}, want exactly the ALWAYS set {sorted(always)} -- an "
                f"unresolved base must under-select to ALWAYS, never to nothing")
        r.ok("an unresolved base degrades to exactly the ALWAYS set (never empty, never "
             "silently wrong)")
    except Exception as e:
        r.fail("SPLIT-E select_touched_tests unresolved base", f"{type(e).__name__}: {e}")


def test_splite_always_touched_files_exist_on_disk(r: SubTestResult):
    print("\n--- SPLIT-E: every ALWAYS-touched path is a real file, and every one is cheap ---")
    # A regression guard for the ALWAYS list itself: a renamed ratchet file would silently
    # drop out of every future `--tier touched` run (pytest would just not collect a
    # nonexistent path) with no red anywhere else, since `select_touched_tests` never checks
    # existence -- this is the one place that does.
    g = _gate()
    try:
        import pathlib
        pkg_root = pathlib.Path(g._PKG)
        missing = [p for _, p in g._ALWAYS_TOUCHED if not (pkg_root / p).is_file()]
        assert not missing, f"ALWAYS-touched path(s) do not exist on disk: {missing}"
        assert len(g._ALWAYS_TOUCHED) == 6, (
            f"expected exactly the 6 named ratchets (docs/citation/mutation/seam/"
            f"skip-budget/floors), found {len(g._ALWAYS_TOUCHED)}")
        r.ok(f"all {len(g._ALWAYS_TOUCHED)} ALWAYS-touched ratchet file(s) exist on disk")
    except Exception as e:
        r.fail("SPLIT-E ALWAYS files exist", f"{type(e).__name__}: {e}")
