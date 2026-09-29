"""Precompile-cache scoping for the compiled tier (split out of `compiled.py`).

Owns the scoping of dynamo's persistent precompile cache (PC-2): the process-global lock
(`_precompile_flag_lock`), the context managers that flip `caching_precompile` for a scope
(`_precompile_off_ctx`, `_precompile_ctx`), the per-fingerprint probe that decides which
scope a program wants (`_wants_precompile_off`), and the attach-failure recovery pair
(`_is_precompile_attach_failure`, `_clear_dynamo_precompile_store`). `compiled.py` re-exports
every name here.

`_wants_precompile_off` reaches back into `compiled.py` lazily, inside the function, so this
module never imports `compiled.py` at module scope and a test that monkeypatches
`compiled._get_or_make_codegen_fn` still sees its own value."""
from __future__ import annotations

import contextlib
import os
import threading
from typing import Any


# COMPILE-51b: `caching_precompile`'s guard-state pickler crashes ("cannot pickle
# '_thread._local'") on the first real compile whose trace needs a resume frame
# across a `torch._dynamo.disable()` boundary -- `_get_gauss_kernels`'s own
# kernel-cache lock (COMPILETRY-50 D2) is such a boundary. Torch-internal (2.12),
# not TEX's own code (see the COMPILE-51 finding writeup). Scoped OFF, in-memory
# only, for the `_has_fn_calls` class `fncalls_compile` tracks; every other
# compiled program keeps disk-persisted `caching_precompile` (PC-2) unchanged.
# `_COMPILE_POOL`/`_WARM_POOL` (both defined in `compiled.py`) run concurrently.
# On torch 2.12 a dynamo config patch is a per-thread override (a ContextVar), so
# a patch on one pool's thread is invisible to the other and the two compiles need
# no serialisation. TEX sets no torch floor, though, and on a build where the patch
# writes a shared module global the two pools' windows would race, so
# `_patch_is_thread_local` checks once which kind of build this is. Only on a shared-global
# build does `_precompile_flag_lock` serialise EVERY scoped window (whichever value it sets)
# against every other one, and it is then held for the whole compile, which can be long.
_precompile_flag_lock = threading.Lock()
_patch_thread_local: bool | None = None


def _patch_is_thread_local(_dc) -> bool:
    """True when `_dc.patch(...)` on one thread is invisible to every other thread.
    Measured once, under `_precompile_flag_lock` so no other scoped window can be open
    while the flag is briefly flipped."""
    global _patch_thread_local
    if _patch_thread_local is None:
        with _precompile_flag_lock:
            if _patch_thread_local is None:
                seen: list = []
                try:
                    original = _dc.caching_precompile
                    with _dc.patch(caching_precompile=not original):
                        probe = threading.Thread(
                            target=lambda: seen.append(_dc.caching_precompile))
                        probe.start()
                        probe.join()
                        _patch_thread_local = (_dc.caching_precompile == (not original)
                                               and bool(seen) and seen[0] == original)
                except Exception:
                    _patch_thread_local = False
    return _patch_thread_local


@contextlib.contextmanager
def _precompile_scoped(_dc, *, caching_precompile: bool):
    """Shared body for both scoped values: the patch, under `_precompile_flag_lock` unless
    the patch is per-thread (see the comment above `_precompile_flag_lock`)."""
    if _patch_is_thread_local(_dc):
        with _dc.patch(caching_precompile=caching_precompile):
            yield
    else:
        with _precompile_flag_lock, _dc.patch(caching_precompile=caching_precompile):
            yield


def _precompile_off_ctx(_dc):
    return _precompile_scoped(_dc, caching_precompile=False)


def _precompile_ctx(*, disable: bool = False):
    """Dynamo's persistent precompile cache (PC-2). `disable=True` (COMPILE-51b)
    scopes `caching_precompile` OFF for the `_has_fn_calls` class instead (see
    module comment). No-op when unsupported."""
    try:
        import torch._dynamo.config as _dc
        if hasattr(_dc, "caching_precompile"):
            return (_precompile_off_ctx(_dc) if disable
                    else _precompile_scoped(_dc, caching_precompile=True))
    except Exception:
        pass
    return contextlib.nullcontext()


def _wants_precompile_off(program, type_map, fingerprint) -> bool:
    """COMPILE-51b: True when this program's codegen fn is `_has_fn_calls`.
    Reuses `_get_or_make_codegen_fn`'s per-fingerprint memo (PC-3): no extra emit."""
    if program is None or type_map is None:
        return False
    try:
        from .compiled import _get_or_make_codegen_fn
        cg_fn = _get_or_make_codegen_fn(program, type_map, fingerprint)
    except Exception:
        return False
    return bool(getattr(cg_fn, '_has_fn_calls', False))


# Error signatures that mean a persisted precompile entry failed to ATTACH
# (stale/corrupt/shape- or version-mismatched) rather than the program being
# genuinely uncompilable. These must NOT blacklist the fingerprint — instead the
# stale dynamo store is cleared so the next run recompiles fresh (PC-2).
def _is_precompile_attach_failure(e: Exception) -> bool:
    name = type(e).__name__
    msg = str(e)
    if name == "AssertionError":
        return True  # guard miss after attach (incl. first-call shape != saved)
    if name == "NameError" and "__compiled_fn_" in msg:
        return True  # same-position source-body edit / stale bytecode
    if "Compile package was created with a different" in msg:
        return True  # torch/CUDA/GPU/triton version mismatch (SystemInfo check)
    return False


def _clear_dynamo_precompile_store() -> None:
    """Delete the persisted dynamo precompile subdir so a poisoned/stale entry
    can't crash every later session (PC-2 recovery).

    HOUSE-50/H3 (TRK-226): this wipes the WHOLE `dynamo/` subdir, not just the one
    fingerprint whose attach just failed — deliberately, confirmed by
    `tests/test_v015_phase1.py::test_pc2_precompile_safety`'s own scope check. Dynamo's
    `caching_precompile` (2.12) exposes no documented per-entry invalidation call and no
    stable, version-independent way to map a TEX cache_key back to the on-disk path it wrote
    under `dynamo/` — guessing at that internal layout risks leaving the ACTUAL poisoned
    entry behind (the failure this function exists to clear would then repeat forever)
    for the sake of sparing entries this function cannot safely identify as unrelated.
    TRK-226 confirmed the resulting cost: under one process that has accumulated many
    OTHER fingerprints' valid, non-stale precompile entries in the SAME shared store, one
    program's genuine stale-attach failure collaterally evicts every one of them, so each
    pays one real recompile on its own next call (measured on real hardware: 5 real
    compiles where 1 was expected, TRK-226's own tracker row). That
    cost is real but BOUNDED and SELF-CORRECTING — every fresh compile after the wipe
    writes a clean entry, so the SAME fingerprint cannot re-trigger this path for the same
    reason twice — and it trades a bounded, one-time-per-fingerprint slow patch for the
    only correctness guarantee available without deeper torch-internals knowledge this
    project does not have today. See DEVELOPMENT.md's "Rejected design decisions" for the
    per-fingerprint-scoped alternative this rejects, and why."""
    try:
        import shutil
        from pathlib import Path
        root = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
        if not root:
            return
        # PC-2 safety: only clear a TEX-OWNED store. A user (or another custom
        # node) may point TORCHINDUCTOR_CACHE_DIR at a shared dir — deleting its
        # dynamo/ would wipe every tool's precompile entries on one TEX failure.
        try:
            from ..tex_cache import get_cache
            owned = get_cache().torch_compile_cache_dir.resolve()
            rp = Path(root).resolve()
            if rp != owned and owned not in rp.parents:
                return  # foreign dir — leave it untouched
        except Exception:
            return  # can't prove ownership → don't delete
        d = os.path.join(root, "dynamo")
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
    except Exception:
        pass
