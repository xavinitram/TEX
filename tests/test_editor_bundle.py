"""The code editor's bundle and the word lists it is built from.

`js/tex_cm6_bundle.js` is built from `editor_build/src` by `npm run build`. Its keyword,
built-in-variable, binding-prefix and stdlib-function lists are generated into
`editor_build/src/tex_lexicon.mjs` by `tools/gen_editor_lexicon.py`, so a new function or
keyword reaches the editor by regenerating and rebuilding. These rows pin that chain:

  - the generated file is current (no node needed);
  - the bundle's own highlighter and completion lists equal the lexer, the type checker and
    the stdlib registry (node, running the shipped bundle);
  - the editor behaviour checks in `tests/js_editor_bundle_checks.cjs` pass (node);
  - when `editor_build/node_modules` exists, a fresh build equals the committed bundle, so a
    source change that was not followed by a rebuild is caught.
"""
import importlib.util
import json
import pathlib
import shutil
import subprocess
import sys

import pytest

_PKG = pathlib.Path(__file__).resolve().parent.parent
_BUNDLE = _PKG / "js" / "tex_cm6_bundle.js"
_BUILD = _PKG / "editor_build"
_NODE = shutil.which("node")
_needs_node = pytest.mark.skipif(_NODE is None, reason="node is not installed")


def _generator():
    key = "_gen_editor_lexicon"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, str(_PKG / "tools" / "gen_editor_lexicon.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


def _lf(text):
    return text.replace("\r\n", "\n")


def test_generated_lexicon_is_current():
    gen = _generator()
    committed = _lf((_BUILD / "src" / "tex_lexicon.mjs").read_text(encoding="utf-8"))
    assert committed == gen.render(gen.build()), \
        "editor_build/src/tex_lexicon.mjs is stale: run tools/gen_editor_lexicon.py, then `npm run build` in editor_build"


def test_lexicon_holds_every_language_word():
    from TEX_Wrangle.tex_compiler.lexer import BINDING_TYPE_PREFIXES, KEYWORDS
    from TEX_Wrangle.tex_compiler.type_checker import _BUILTIN_VAR_NAMES
    from TEX_Wrangle.tex_runtime import stdlib_registry as R
    from TEX_Wrangle.tex_runtime.stdlib import TEXStdlib  # noqa: F401  (populates REGISTRY)
    data = _generator().build()
    assert set(data["LEX_KEYWORDS"]) == set(KEYWORDS)
    assert set(data["LEX_BINDING_PREFIXES"]) == set(BINDING_TYPE_PREFIXES)
    assert set(data["LEX_COORD_VARS"]) | set(data["LEX_CONSTANTS"]) == set(_BUILTIN_VAR_NAMES)
    assert {f[0] for f in data["LEX_FUNCTIONS"]} == {e.name for e in R.REGISTRY}
    assert {a for a, _ in data["LEX_ALIASES"]} == {a for e in R.REGISTRY for a in e.aliases}


@_needs_node
def test_shipped_bundle_lists_equal_the_language():
    from TEX_Wrangle.tex_compiler.lexer import KEYWORDS
    from TEX_Wrangle.tex_compiler.type_checker import _BUILTIN_VAR_NAMES
    from TEX_Wrangle.tex_runtime import stdlib_registry as R
    from TEX_Wrangle.tex_runtime.stdlib import TEXStdlib  # noqa: F401
    proc = subprocess.run([_NODE, str(_PKG / "tests" / "js_editor_bundle_checks.cjs"), "--dump"],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    got = json.loads(proc.stdout.strip().splitlines()[-1])
    functions = {e.name for e in R.REGISTRY} | {a for e in R.REGISTRY for a in e.aliases}
    constants = {"PI", "TAU", "E"}
    assert set(got["keywords"]) == set(KEYWORDS)
    assert set(got["builtins"]) == functions
    assert set(got["constants"]) == constants
    assert set(got["coordVars"]) == set(_BUILTIN_VAR_NAMES) - constants
    # Every completion is a word of the language, and every word of the language completes.
    assert set(got["labels"]) == functions | set(KEYWORDS) | set(_BUILTIN_VAR_NAMES)


@_needs_node
def test_editor_bundle_behaviour_under_node():
    proc = subprocess.run([_NODE, str(_PKG / "tests" / "js_editor_bundle_checks.cjs")],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr


@_needs_node
@pytest.mark.skipif(not (_BUILD / "node_modules" / "rollup").is_dir(),
                    reason="editor_build/node_modules is not installed (npm ci)")
def test_committed_bundle_is_a_fresh_build_of_the_sources(tmp_path):
    out = tmp_path / "bundle.js"
    proc = subprocess.run(
        [_NODE, str(_BUILD / "node_modules" / "rollup" / "dist" / "bin" / "rollup"),
         "-c", "--file", str(out)],
        cwd=str(_BUILD), capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _lf(out.read_text(encoding="utf-8")) == _lf(_BUNDLE.read_text(encoding="utf-8")), \
        "js/tex_cm6_bundle.js is not the build of editor_build/src: run `npm run build` in editor_build"
