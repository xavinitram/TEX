"""
STR-7 (cluster 3) — codegen persistence seam.

Generated-function persistence: pseudo-filename minting, a bounded linecache
registration so tracebacks in generated code resolve, K1's paired synthetic-module
registration (below) so a Dynamo graph-break resume resolves too, and marshal-based
rematerialization of a cached code object. Free functions with their own bounded
linecache state; zero `_CodeGen` reference => a strict leaf (codegen.py and
tex_cache import back its entry points). Contract unchanged: the CALLER
validates version/MAGIC/SHA before `materialize_codegen`.
"""
from __future__ import annotations
from typing import Any

import linecache as _linecache
import sys as _sys
import types as _types
from collections import deque as _deque

# Bounded set of registered codegen pseudo-filenames, pruned oldest-first so
# linecache (and, K1, sys.modules) can't grow without bound across a long session.
_LINECACHE_KEYS: "_deque[str]" = _deque()
_LINECACHE_MAX = 64


def _cg_filename(fingerprint: str, cancel: bool = False) -> str:
    """The pseudo-filename for a fingerprinted codegen module — must match
    between build() and materialize_codegen() so linecache keys line up. The
    cancel-polling build emits different source, so it gets its own name."""
    return f"<tex_codegen_{fingerprint[:16]}{'_ck' if cancel else ''}>"


def _codegen_module_name(filename: str) -> str:
    """K1 (v0.50.0 Phase C, F2/B3#1): the sys.modules key paired 1:1 with a codegen
    pseudo-filename — stripped of the angle brackets `_cg_filename` wraps it in, since
    `__name__`/a sys.modules key is not a display string. Same key as the linecache
    registration it is evicted alongside (below), by construction."""
    return "tex_codegen_module_" + filename.strip("<>")


def _codegen_exec_namespace(filename: str, seed: dict) -> dict:
    """K1 (v0.50.0 Phase C, F2 + B3#1): the exec-time namespace for a codegen build — a
    REAL module's own `__dict__`, registered in `sys.modules`, not a bare dict.

    THE DEFECT: the namespace `_CodeGen.build()` / `materialize_codegen()` used
    to run generated code in had no `__name__` (only `_MF`/`_CK`/`_SCM`) — not shaped like a real
    module's globals at all. A graph break inside compiled generated code needs Dynamo
    to build a resume continuation (`create_resume`) or resolve a traced frame's globals
    module (`get_globals_source_and_value`), both of which read `f_globals["__name__"]`
    and then resolve it via `importlib.import_module` — which returns `sys.modules[name]`
    directly, with no finder, whenever `name` is already present. A bare synthetic string
    (present nowhere in `sys.modules`) only trades the `KeyError` for a
    `ModuleNotFoundError` at that resolution step (confirmed by running, B3#1) — the
    dunder must resolve to something REAL.

    THE FIX IS NOT "hand it a live TEX module's `__name__`": Dynamo's `CompilePackage`
    installs resume-function globals directly onto `sys.modules[name].__dict__`, and
    never fully undoes it — pointing every codegen build at one shared, live product
    module would leak an unbounded, cross-thread-mutated set of attributes onto it for
    the process lifetime (B3#1). So each codegen build gets its OWN dedicated,
    disposable `types.ModuleType`, and this function returns that module's `__dict__`
    itself (not a separate dict later assigned) so any global write Dynamo installs
    lands on the one object `sys.modules[name]` — this module and the traced frame's own
    globals — see the same object, exactly as a real module's do.

    BOUNDED: registered and evicted together with the paired linecache entry
    (`_register_codegen_linecache`, below, shares the SAME key) — a long session's oldest
    builds release the placeholder module, and whatever Dynamo attached to it, instead of
    accumulating forever. Deterministic filenames repeat (the same fingerprint recompiled
    on a warm cache path): the existing module for that name is reused so a resume built
    against an EARLIER call's frame still finds the SAME module/dict it was built
    against — a fresh `ModuleType` every call would silently break that resume the moment
    a second build for the same filename replaced `sys.modules[name]` out from under it.
    """
    name = _codegen_module_name(filename)
    mod = _sys.modules.get(name)
    if mod is None:
        mod = _types.ModuleType(name)
        mod.__file__ = filename
        _sys.modules[name] = mod
    mod.__dict__.update(seed)
    return mod.__dict__


def _register_codegen_linecache(filename: str, src: str) -> None:
    """Register generated source with linecache (for diagnostics / getsource),
    pruning the oldest entry when over the cap. Deterministic filenames repeat: the
    newest source wins (a rebuild can emit different text) and a key is queued once.

    K1: the paired sys.modules entry (`_codegen_exec_namespace`, above) is evicted
    alongside the linecache entry it shares a key with, so the two registries can
    never drift — one always outlives the other by construction, never by omission."""
    lines = src.splitlines(True)
    entry = _linecache.cache.get(filename)
    if entry is not None and entry[2] == lines:
        return
    _linecache.cache[filename] = (len(src), None, lines, filename)
    if filename not in _LINECACHE_KEYS:
        _LINECACHE_KEYS.append(filename)
    while len(_LINECACHE_KEYS) > _LINECACHE_MAX:
        evicted = _LINECACHE_KEYS.popleft()
        _linecache.cache.pop(evicted, None)
        _sys.modules.pop(_codegen_module_name(evicted), None)


def materialize_codegen(blob: bytes, src: str, has_fn_calls: bool,
                        fingerprint: str) -> Any:
    """Rebuild a codegen fn from a marshalled MODULE code object (PC-3).

    The caller is responsible for validating the blob (version/MAGIC/SHA) before
    calling this — marshal.loads on corrupted bytes can hard-crash the process.
    """
    import marshal
    from . import masked_flow as _masked_flow_mod
    code_obj = marshal.loads(blob)
    filename = _cg_filename(fingerprint)
    _register_codegen_linecache(filename, src)
    # LANG-L5: the same global `_CodeGen.build` seeds. A persisted `0.25` program's code
    # object references `_MF`, so a rematerialized one must find it or the sidecar would
    # be a NameError instead of a cook — and it must be the SAME module object, not a
    # re-import with its own state, which the function-local import here guarantees.
    # SCALE-CG-48: `_SCM` (`_scale_pixel_arg`) is the analogous unconditional-per-call
    # global — unlike `_CK` (only referenced when `emit_cancel_polls=True`, which this
    # PC-3 disk-persisted default-route blob never sets, so it needs no entry here), any
    # persisted program with a `pixel_args=` call references `_SCM` on EVERY call, cancel
    # or not, so a rematerialized one needs it exactly as it needs `_MF`. Imported from
    # `.stdlib` (a leaf), never `.codegen`, to keep this module's own "zero `_CodeGen`
    # reference" contract (this file's own docstring) intact.
    # TRK-236: `_THS` (`_stage_codegen_param`) is the same unconditional-per-call shape as
    # `_SCM` — any persisted program with a `$param` reference calls it on every
    # invocation (`_get_param_local`'s preamble), so a rematerialized one needs it too.
    from .stdlib import _scale_pixel_arg, _stage_codegen_param
    # K1: a real, registered module's __dict__ -- see _codegen_exec_namespace's own
    # docstring; this rematerialization path is the second of B3#1's two independent
    # generated-code run sites (the warm-restart marshal path), fixed the same way build() is.
    namespace = _codegen_exec_namespace(
        filename, {"_MF": _masked_flow_mod, "_SCM": _scale_pixel_arg,
                  "_THS": _stage_codegen_param})
    exec(code_obj, namespace)
    fn = namespace["_tex_fn"]
    fn._has_fn_calls = has_fn_calls
    fn._tex_code = code_obj
    fn._tex_src = src
    return fn
