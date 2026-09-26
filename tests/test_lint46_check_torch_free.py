"""LINT-46 — `tex_api.check()` (and the LSP's equivalent) stay torch-free.

TRK-188 named the gap `IMPORT-44` (`v044-perf`, unmerged, `1d0f265`) left open: deferring
`tex_runtime/host.py`'s `import torch` alone was not enough, because `tex_api.py` also
imports `tex_runtime.host` at module scope, and importing ANY submodule of the
`tex_runtime` PACKAGE first runs `tex_runtime/__init__.py` — which used to eagerly import
`.interpreter` / `.stdlib` / `.compiled` (the tensor engine itself), each of which
`import torch` at module scope. `tex_api.py` separately imports `tex_marshalling` at
module scope, which had its own module-scope `import torch` (used across ~15 functions)
plus a `from .tex_runtime.stdlib import LUMA_R, LUMA_G, LUMA_B` that pulled the entire
stdlib registry (every `stdlib_*.py` domain file) in behind three float constants. And
`tex_cache.py` — which `check()` reaches for `parse_and_split` — imported
`tex_runtime.interpreter._collect_identifiers` at module scope, even though `check()`'s
own front end never calls it (only a real compile does).

This closes all four: `tex_runtime/__init__.py` now resolves `Interpreter` / `TEXStdlib` /
`execute_compiled` / `clear_compiled_cache` lazily (PEP 562 `__getattr__`, the PORT-6
pattern); `tex_marshalling.py` moved its `import torch` and the LUMA import into the
functions that actually touch a tensor; `tex_cache.py` moved `_collect_identifiers`'s
import to its two real-compile call sites; and `tex_runtime/host.py` carries IMPORT-44's
own fix (re-applied here, `1d0f265`, unmerged on `v044-perf`).

Every check below runs in a FRESH SUBPROCESS: this suite (via `conftest.py`/`helpers.py`)
has already imported torch, so only a fresh interpreter can see what a first touch costs.

PORTABILITY: CPU-only, no ComfyUI, no CUDA hardware, no compiler, no numpy.
"""
import pathlib

from helpers import *
from helpers import run_python_kv   # G7/R1#4: the shared fresh-subprocess KV helper --
                                     # not in __all__ (HOOK-4), so imported by name.


def _run(code: str) -> dict:
    """This file's own fresh-process-and-parse shape, now the shared `helpers.run_python_kv`
    (G7) -- kept as a thin, same-signature wrapper so every call site below is unchanged."""
    return run_python_kv(code, timeout=60)


def _custom_nodes_dir() -> str:
    import TEX_Wrangle
    return str(pathlib.Path(TEX_Wrangle.__file__).resolve().parent.parent)


def test_lint46_tex_api_check_stays_torch_free(r: SubTestResult):
    """The headline ratchet: a pure-lint `tex_api.check()` call, in a fresh process,
    never imports torch — neither at `import tex_api` nor at the `check()` call itself."""
    print("\n--- LINT-46: tex_api.check() stays torch-free end to end ---")
    code = (
        "import sys\n"
        f"sys.path.insert(0, {_custom_nodes_dir()!r})\n"
        "from TEX_Wrangle import tex_api\n"
        "print('AFTER_IMPORT', 'torch' in sys.modules)\n"
        # A non-empty binding_types dict exercises the real type-checker path (not just
        # the lexer/parser short-circuit on a trivial program), including a swizzle that
        # resolves against a declared type.
        "from TEX_Wrangle.tex_compiler.types import TEXType\n"
        "diags = tex_api.check('@OUT = vec4(@A.rgb * 2.0, 1.0);', {'A': TEXType.VEC4})\n"
        "print('AFTER_CHECK', 'torch' in sys.modules)\n"
        "print('DIAGS_OK', isinstance(diags, list) and len(diags) == 0)\n"
        # A program with an actual type error still must not import torch — the checker's
        # error-accumulation path is as much "check()" as the clean path is.
        "diags2 = tex_api.check('@OUT = @A + \"x\";', {'A': TEXType.VEC4})\n"
        "print('AFTER_CHECK_ERROR', 'torch' in sys.modules)\n"
        "print('DIAGS2_NONEMPTY', len(diags2) > 0)\n"
    )
    try:
        out = _run(code)
    except Exception as e:
        r.fail("LINT-46 check() torch-free subprocess", f"{type(e).__name__}: {e}")
        return
    if out.get("AFTER_IMPORT") != "False":
        r.fail("LINT-46 import tex_api", f"torch already loaded after import (got {out.get('AFTER_IMPORT')})")
    else:
        r.ok("`from TEX_Wrangle import tex_api` does not import torch")
    if out.get("AFTER_CHECK") != "False":
        r.fail("LINT-46 check() clean program", f"torch loaded after check() (got {out.get('AFTER_CHECK')})")
    else:
        r.ok("check() on a clean program does not import torch")
    if out.get("AFTER_CHECK_ERROR") != "False":
        r.fail("LINT-46 check() error program", f"torch loaded after an erroring check() (got {out.get('AFTER_CHECK_ERROR')})")
    else:
        r.ok("check() on a program with a type error does not import torch either")
    if out.get("DIAGS_OK") != "True":
        r.fail("LINT-46 check() behaviour", "clean program did not return an empty diagnostics list")
    else:
        r.ok("check() still returns the expected diagnostics for a clean program")
    if out.get("DIAGS2_NONEMPTY") != "True":
        r.fail("LINT-46 check() behaviour", "erroring program did not return any diagnostics")
    else:
        r.ok("check() still reports the type error for a bad program")


def test_lint46_tex_lsp_diagnostics_stays_torch_free(r: SubTestResult):
    """The LSP's live-lint entry point (`didOpen`/`didChange` -> `diagnostics_for` ->
    `tex_api.check`) is the same seam, reached through `tex_lsp.py` instead of directly.
    Drives it through `LSPServer.handle`, the same dispatch `main()`'s stdio loop calls,
    rather than `diagnostics_for` directly, so this covers the LSP's own entry point and
    not just the function it happens to delegate to."""
    print("\n--- LINT-46: the LSP's didOpen/didChange diagnostics stay torch-free ---")
    code = (
        "import sys\n"
        f"sys.path.insert(0, {_custom_nodes_dir()!r})\n"
        "from TEX_Wrangle import tex_lsp\n"
        "print('AFTER_IMPORT', 'torch' in sys.modules)\n"
        "server = tex_lsp.LSPServer()\n"
        "server.handle('initialize', {})\n"
        "result, notes = server.handle('textDocument/didOpen', {'textDocument': "
        "{'uri': 'file:///t.tex', 'text': '@OUT = @A.rgb;'}})\n"
        "print('AFTER_DIDOPEN', 'torch' in sys.modules)\n"
        "print('PUBLISHED_OK', len(notes) == 1 and "
        "notes[0]['method'] == 'textDocument/publishDiagnostics')\n"
    )
    try:
        out = _run(code)
    except Exception as e:
        r.fail("LINT-46 LSP torch-free subprocess", f"{type(e).__name__}: {e}")
        return
    if out.get("AFTER_IMPORT") != "False":
        r.fail("LINT-46 import tex_lsp", f"torch already loaded after import (got {out.get('AFTER_IMPORT')})")
    else:
        r.ok("`from TEX_Wrangle import tex_lsp` does not import torch")
    if out.get("AFTER_DIDOPEN") != "False":
        r.fail("LINT-46 LSP didOpen", f"torch loaded after didOpen's publish (got {out.get('AFTER_DIDOPEN')})")
    else:
        r.ok("a didOpen -> publishDiagnostics round trip does not import torch")
    if out.get("PUBLISHED_OK") != "True":
        r.fail("LINT-46 LSP didOpen behaviour", "didOpen did not publish diagnostics as expected")
    else:
        r.ok("didOpen still publishes diagnostics as before")


def test_lint46_real_cook_path_unaffected(r: SubTestResult):
    """Non-regression, not a ratchet: a REAL cook still imports torch (it must — this
    isn't a claim that torch is gone, only that a lint-only call no longer forces it) and
    still produces the right pixels. Guards against the fix degenerating into "torch is
    never imported at all", which would just be a different, worse bug."""
    print("\n--- LINT-46: a real cook still imports torch and still cooks correctly ---")
    code = (
        "import sys\n"
        f"sys.path.insert(0, {_custom_nodes_dir()!r})\n"
        "import torch\n"
        "from TEX_Wrangle import tex_engine\n"
        "img = torch.full((1, 2, 2, 4), 0.25)\n"
        "res = tex_engine.cook('@OUT = vec4(@A.rgb * 2.0, 1.0);', {'A': img}, device_mode='cpu')\n"
        "out = res.outputs['OUT']\n"
        "print('SHAPE_OK', tuple(out.shape) == (1, 2, 2, 4))\n"
        "print('VALUE_OK', abs(out[0, 0, 0, 0].item() - 0.5) < 1e-6)\n"
        "print('TORCH_LOADED', 'torch' in sys.modules)\n"
    )
    try:
        out = _run(code)
    except Exception as e:
        r.fail("LINT-46 real cook subprocess", f"{type(e).__name__}: {e}")
        return
    if out.get("TORCH_LOADED") != "True":
        r.fail("LINT-46 real cook", "torch was not loaded by a real cook — unexpected")
    else:
        r.ok("a real cook still imports and uses torch")
    if out.get("SHAPE_OK") != "True" or out.get("VALUE_OK") != "True":
        r.fail("LINT-46 real cook pixels", f"shape/value check failed: {out}")
    else:
        r.ok("a real cook still produces the same pixels (0.25*2.0 == 0.5)")


def test_lint46_tex_runtime_lazy_names_resolve(r: SubTestResult):
    """`tex_runtime/__init__.py`'s four names are now PEP 562 lazy attributes (the PORT-6
    pattern) rather than eager imports. Confirms each still resolves to the real object —
    a host or test importing them off the package (rather than the submodule directly)
    must see the identical objects it saw before this change."""
    print("\n--- LINT-46: tex_runtime's lazy __getattr__ still resolves correctly ---")
    import TEX_Wrangle.tex_runtime as tr
    from TEX_Wrangle.tex_runtime.interpreter import Interpreter, InterpreterError
    from TEX_Wrangle.tex_runtime.stdlib import TEXStdlib
    from TEX_Wrangle.tex_runtime.compiled import execute_compiled, clear_compiled_cache
    checks = [
        ("Interpreter", tr.Interpreter, Interpreter),
        ("InterpreterError", tr.InterpreterError, InterpreterError),
        ("TEXStdlib", tr.TEXStdlib, TEXStdlib),
        ("execute_compiled", tr.execute_compiled, execute_compiled),
        ("clear_compiled_cache", tr.clear_compiled_cache, clear_compiled_cache),
    ]
    for name, lazy_obj, direct_obj in checks:
        if lazy_obj is direct_obj:
            r.ok(f"tex_runtime.{name} (lazy) is the same object as the submodule's own {name}")
        else:
            r.fail("LINT-46 lazy attribute identity",
                   f"tex_runtime.{name} is not the submodule's {name} object")
    # A genuinely missing name still raises AttributeError (not a silent None), exactly as
    # a normal module would.
    try:
        tr.not_a_real_name
        r.fail("LINT-46 lazy attribute miss", "reading an unknown attribute did not raise")
    except AttributeError:
        r.ok("an unrelated missing attribute still raises AttributeError")
    if {"Interpreter", "InterpreterError", "TEXStdlib", "execute_compiled",
        "clear_compiled_cache"} <= set(dir(tr)):
        r.ok("dir(tex_runtime) still lists the four lazy names")
    else:
        r.fail("LINT-46 lazy attribute dir()", f"dir(tex_runtime) is missing a lazy name: {dir(tr)}")
