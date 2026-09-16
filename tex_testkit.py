"""TEX_Wrangle.tex_testkit — the state-isolation kit, for a host that cannot reach into tests/.

HOOK-4: `cold_engine_state` and its neighbours were reachable only by loading
`tests/helpers.py` off disk under a private module name, because TEX's own suite reaches them
with a bare `from helpers import *` — which needs `tests/` on `sys.path`, and putting an
embedding host's own `tests/` there risks shadowing that host's `tests` package. This module is
the sibling-module shape, chosen over a `tests/__init__.py` (which could move TEX's own pytest
rootdir/collection). `tests/helpers.py` re-exports these three names so `from helpers import *`
is unchanged for TEX's own suite; an embedding host imports this module instead.

The three make up one named "state-isolation kit": `make_img` (a deterministic test image
factory), `cold_engine_state` (a scratch `TEX_CACHE_DIR` + clean warm-state table for a block,
restored exactly), `armed_profiler` (PROF-1 armed on a clean table, always disarmed).

Stability (`DEVELOPMENT.md` §"API stability tiers (ENG-5)"): not a row in that table — ENG-5
tiers the PRODUCTION embedding surface (`tex_api`/`tex_engine`/`tex_cli`), and no ComfyUI cook
ever reaches this module. Read against that table's own vocabulary, this sits closest to
**Tier 2 — Semi**: additive-only, and the exposed name set is canary-pinned
(`tests/test_hook4_testkit.py`), so dropping or renaming one is a release-note decision, not a
silent break. It is deliberately NOT Tier 1 (no production host embeds a test kit) and NOT
Tier 3's "no promise, import at your own risk" either — HOOK-4 exists precisely so this one
surface carries a promise without a host having to path-load a test file to get one.

Costs only what `tests/helpers.py` already costs to expose these three: stdlib (`os`, `shutil`,
`tempfile`) and `torch` at module scope, nothing more — the compiler/interpreter/codegen imports
`tests/helpers.py` also carries belong to its OTHER helpers, never promoted here. No `pytest`
import. Not reached from the ComfyUI adapter files (`tex_node.py`, `__init__.py`,
`tex_runtime/host.py`).
"""
from __future__ import annotations

import os
import shutil
import tempfile

import torch

__all__ = ["make_img", "cold_engine_state", "armed_profiler"]


def make_img(B=1, H=8, W=8, C=3, seed=42) -> torch.Tensor:
    """Deterministic test image [B,H,W,C]."""
    torch.manual_seed(seed)
    return torch.rand(B, H, W, C)


class cold_engine_state:
    """A scratch `TEX_CACHE_DIR` + a clean warm-state/verdict table for the duration of a block,
    restored exactly on the way out.

    Four v0.31 tests needed this and three had hand-rolled it, which is a real hazard rather
    than a tidiness one: `run_all.py` runs the whole suite in ONE process, in order, so a single
    missed restore leaks a deleted cache directory into every later test. One implementation,
    one teardown.

    It is also load-bearing for correctness in at least one place: a cache probe checks memory
    and then DISK, so a test asserting "this program compiles" has to start from a cache that
    has never seen it — otherwise it passes vacuously on the second run of the suite.

        with cold_engine_state():
            ...                       # a fresh cache dir; warm state and memo start empty

    `warm=True` (the default) also clears `graphed._capturable_memo`, `profile._STATE` and
    `warm_state`'s load latch + path/tag memos; pass False when only the program cache matters.

    PROF-1's cost table belongs in that list even though it is not "warm state" in the CACHE-3
    sense: it is process-global engine state this release adds, it is keyed by fingerprints that
    a scratch cache dir invalidates, and leaving it out is exactly the leak class this fixture
    was written to end."""

    def __init__(self, *, warm: bool = True):
        self.warm = warm
        self.dir = None

    def __enter__(self):
        from TEX_Wrangle import tex_cache
        from TEX_Wrangle.tex_runtime import graphed, warm_state, profile, autotier
        self._cache_mod, self._graphed, self._ws = tex_cache, graphed, warm_state
        self._prof, self._autotier = profile, autotier
        self.dir = tempfile.mkdtemp(prefix="tex_cold_")
        self._prev_env = os.environ.get("TEX_CACHE_DIR")
        self._prev_cache = tex_cache._cache_instance
        os.environ["TEX_CACHE_DIR"] = self.dir
        tex_cache._cache_instance = None
        if self.warm:
            self._prev_memo = dict(graphed._capturable_memo)
            graphed._capturable_memo.clear()
            warm_state._reset_for_test()
            profile.reset()
            autotier._reset_for_test()
        return self

    def __exit__(self, *exc):
        if self._prev_env is None:
            os.environ.pop("TEX_CACHE_DIR", None)
        else:
            os.environ["TEX_CACHE_DIR"] = self._prev_env
        self._cache_mod._cache_instance = self._prev_cache
        if self.warm:
            self._graphed._capturable_memo.clear()
            self._graphed._capturable_memo.update(self._prev_memo)
            self._ws._reset_for_test()
            self._prof.reset()
            self._autotier._reset_for_test()
        shutil.rmtree(self.dir, ignore_errors=True)
        return False


def armed_profiler():
    """Arm PROF-1 on a clean table, and ALWAYS disarm — a leaked `enable()` puts a CUDA sync
    into every later test in the suite, and `run_all.py` runs the whole suite in one process.

    Promoted from `test_v031_phase2._armed` when a second release needed it: two copies of a
    "must always run the finally" fixture is one copy away from a leak nobody notices, because
    the symptom (everything after it gets slower) does not look like a failure.

        with armed_profiler() as P:
            ...
    """
    import contextlib

    from TEX_Wrangle.tex_runtime import profile as _P

    @contextlib.contextmanager
    def _cm():
        _P.reset()
        _P.enable()
        try:
            yield _P
        finally:
            _P.disable()
            _P.reset()

    return _cm()
