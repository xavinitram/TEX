"""COMPILE-51b precompile-scoping helpers — SPLIT (v0.51 Phase C, FIX-COMPILE51 C0).

Split mechanically out of `compiled.py` (the SPLIT-47/K0 pattern: every body below is
byte-identical to the code it replaced there at the moment of the move — AGENTS.md
§"Trades to REFUSE", mechanical moves only, never an "improvement" mid-move).
`compiled.py` was at 1999/2000 lines with no headroom floor (REG-2, B4#3, R4#2) and
COMPILE-51b's own fix (see below) was about to touch this exact code, so it moves out
FIRST, unchanged, before the fix lands — following the same shape this module's siblings
(`compiled_capability.py`, `compiled_exec_support.py`, `compiled_promotion.py`) already
document.

This is C0's move only: every body below is unchanged from `compiled.py` at `365fdb4`
(the base this fix lane starts from) -- C1 (the next commit) is where the lock's actual
coverage changes.

This module owns dynamo's persistent precompile cache (PC-2) scoping: the process-global
lock (`_precompile_flag_lock`), the two context-manager shapes that flip
`caching_precompile` for a scope (`_precompile_off_ctx`, `_precompile_ctx`), the
per-fingerprint probe that decides which scope a program wants
(`_wants_precompile_off`), and the attach-failure recovery pair
(`_is_precompile_attach_failure`, `_clear_dynamo_precompile_store`). `compiled.py` imports
this module at its own top level and re-exports every name below, so `compiled.NAME` and
`from .compiled import NAME` keep resolving unchanged for every external caller (tests
included) — the ROUTE-45 shape SPLIT-E used.

This module reaches back into `compiled.py` for the one name that stays there
(`_get_or_make_codegen_fn`) lazily, inside the one function that needs it
(`_wants_precompile_off`) — the same posture `compiled_capability.py` already uses for
`compiled._backend_status`/`compiled._setup_msvc_env` — so this module never imports
`compiled.py` at its own module scope (there is no load-time cycle) AND a test that
monkeypatches `compiled._get_or_make_codegen_fn` (see
`tests/test_compile51b_precompile_disable.py`) still observes its own patched value: the
`from .compiled import _get_or_make_codegen_fn` below is re-evaluated on every call, not
bound once at import time."""
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
# The flag is process-global while `_COMPILE_POOL`/`_WARM_POOL` run concurrently,
# so `_precompile_flag_lock` serializes an off-scoped window against either pool.
_precompile_flag_lock = threading.Lock()


@contextlib.contextmanager
def _precompile_off_ctx(_dc):
    with _precompile_flag_lock, _dc.patch(caching_precompile=False):
        yield


def _precompile_ctx(*, disable: bool = False):
    """Dynamo's persistent precompile cache (PC-2). `disable=True` (COMPILE-51b)
    scopes `caching_precompile` OFF for the `_has_fn_calls` class instead (see
    module comment). No-op when unsupported."""
    try:
        import torch._dynamo.config as _dc
        if hasattr(_dc, "caching_precompile"):
            return _precompile_off_ctx(_dc) if disable else _dc.patch(caching_precompile=True)
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
