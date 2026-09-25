# TEX Runtime — tensor interpreter, stdlib, and optional compiled execution
#
# LINT-46: these four names ARE the tensor engine — `.interpreter`/`.stdlib`/`.compiled`
# each `import torch` at module scope (MEASURE-44 §3). They used to load eagerly here,
# which meant importing ANY submodule of this package (`tex_api.py` reaches `.host` for
# `CookCancelled`/`CancelToken`) ran this file first and forced the whole engine in behind
# it — even for a pure-lint `tex_api.check()` call that never cooks a pixel. Nothing in
# this repository imports these four names off the PACKAGE (every real call site already
# does `from .tex_runtime.interpreter import Interpreter` etc., naming the submodule), so
# building them on first access costs the eager path nothing and saves the lint path
# everything. PEP 562 module `__getattr__` (the PORT-6 pattern the package root already
# uses for the ComfyUI adapter) resolves each name once and caches it in globals(), so
# `__getattr__` does not fire again for that name.
__all__ = [
    "Interpreter",
    "InterpreterError",
    "TEXStdlib",
    "execute_compiled",
    "clear_compiled_cache",
]

_LAZY_NAMES = frozenset(__all__)


def __getattr__(name):
    """Resolve one of the four engine names lazily. Any other missing name raises
    AttributeError as usual, so `hasattr` on an unrelated name never loads the engine."""
    if name not in _LAZY_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name in ("Interpreter", "InterpreterError"):
        from .interpreter import Interpreter, InterpreterError
        globals()["Interpreter"] = Interpreter
        globals()["InterpreterError"] = InterpreterError
    elif name == "TEXStdlib":
        from .stdlib import TEXStdlib
        globals()["TEXStdlib"] = TEXStdlib
    else:  # execute_compiled / clear_compiled_cache
        from .compiled import execute_compiled, clear_compiled_cache
        globals()["execute_compiled"] = execute_compiled
        globals()["clear_compiled_cache"] = clear_compiled_cache
    return globals()[name]


def __dir__():
    """Keep the lazy names discoverable (`dir()`, tab-completion, doc tools) whether or
    not the engine has been loaded yet."""
    return sorted(set(globals()) | _LAZY_NAMES)
