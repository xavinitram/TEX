"""
STR-7 (cluster 3) — codegen persistence seam.

Generated-function persistence: pseudo-filename minting, a bounded linecache
registration so tracebacks in generated code resolve, and marshal-based
rematerialization of a cached code object. Free functions with their own bounded
linecache state; zero `_CodeGen` reference => a strict leaf (codegen.py and
tex_cache import back the three entry points). Contract unchanged: the CALLER
validates version/MAGIC/SHA before `materialize_codegen`.
"""
from __future__ import annotations
from typing import Any

import linecache as _linecache
from collections import deque as _deque

# Bounded set of registered codegen pseudo-filenames, pruned oldest-first so
# linecache can't grow without bound across a long session.
_LINECACHE_KEYS: "_deque[str]" = _deque()
_LINECACHE_MAX = 64


def _cg_filename(fingerprint: str) -> str:
    """The pseudo-filename for a fingerprinted codegen module — must match
    between build() and materialize_codegen() so linecache keys line up."""
    return f"<tex_codegen_{fingerprint[:16]}>"


def _register_codegen_linecache(filename: str, src: str) -> None:
    """Register generated source with linecache (for diagnostics / getsource),
    pruning the oldest entry when over the cap. Deterministic filenames repeat,
    so skip re-registering (and re-queuing) an already-present key."""
    if filename in _linecache.cache:
        return
    _linecache.cache[filename] = (len(src), None, src.splitlines(True), filename)
    _LINECACHE_KEYS.append(filename)
    while len(_LINECACHE_KEYS) > _LINECACHE_MAX:
        _linecache.cache.pop(_LINECACHE_KEYS.popleft(), None)


def materialize_codegen(blob: bytes, src: str, has_fn_calls: bool,
                        fingerprint: str) -> Any:
    """Rebuild a codegen fn from a marshalled MODULE code object (PC-3).

    The caller is responsible for validating the blob (version/MAGIC/SHA) before
    calling this — marshal.loads on corrupted bytes can hard-crash the process.
    """
    import marshal
    code_obj = marshal.loads(blob)
    filename = _cg_filename(fingerprint)
    _register_codegen_linecache(filename, src)
    namespace: dict[str, Any] = {}
    exec(code_obj, namespace)
    fn = namespace["_tex_fn"]
    fn._has_fn_calls = has_fn_calls
    fn._tex_code = code_obj
    fn._tex_src = src
    return fn
