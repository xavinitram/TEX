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

`FakeCudaEvent` and `DeviceSpy` (below) are deliberately NOT in `__all__` and not part of the
"state-isolation kit" or its stability promise — the same reason `load_counts_harness` sits
outside it in `tests/helpers.py`: their callers import them by name
(`from TEX_Wrangle.tex_testkit import FakeCudaEvent, DeviceSpy`), which needs no entry here and
asks nothing of the canary test that pins the kit's three names. F6 (v0.46.2 Phase C): one
shared mocked-`torch.cuda` scaffold, replacing three that had already drifted apart (R1#1).
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
        from TEX_Wrangle.tex_runtime import graphed, warm_state, profile, autotier, compiled
        self._cache_mod, self._graphed, self._ws = tex_cache, graphed, warm_state
        self._prof, self._autotier, self._compiled = profile, autotier, compiled
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
        if self.warm:
            # C2 (v0.46 Phase C, B5#1): drain BEFORE restoring anything else, so a warm
            # job still in flight when THIS block ends cannot fire its call into
            # `_invoke_cg`/`_params_on_device` (or any other module-level seam a test
            # inside this block monkeypatched) after the block's own patches are already
            # gone — or, worse, during a LATER block's own `cold_engine_state`, where it
            # would silently inflate that later block's own spy counts on the very same
            # seams (measured: a 1-in-3 flake in
            # `test_codegen_param_placement_learned_once`'s learned-once pin).
            #
            # Deliberately EXIT-side only, not also on __enter__: draining on entry too
            # was tried and measured WORSE — it frees the (single-worker) background pool
            # right before this block's OWN first submission, and for a program with no
            # real torch.compile cost (the codegen-only eager adapter) that pool being
            # instantly available let ITS OWN warm job complete and call the STILL-
            # INSTALLED spy before this block's own `finally` restored it, turning an
            # occasional cross-test leak into a deterministic same-block miscount. Exit
            # already drains every block that used this fixture, so by the time the NEXT
            # block's __enter__ runs there is nothing left over to catch anyway — the
            # entry-side call bought no additional safety, only a new race.
            self._compiled._drain_bg_for_test()
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


class FakeCudaEvent:
    """A stand-in for `torch.cuda.Event`, covering every shape this tree's mocked CUDA tests
    needed before F6 in ONE class instead of three that had drifted apart (R1#1 of the
    v0.46.2 Phase C reuse review): pacing's ring constructs events with `Event(blocking=True)`
    and reads `record()`/`synchronize()` call counts and a live-construction counter; the
    profiler constructs them with `Event(enable_timing=True)` and reads `elapsed_time()`
    between two of them, plus `query()` to hold a sample "still running" and flip it later.
    `elapsed_time` is pure arithmetic on a monotonically increasing tick stamped by `record()`
    — never a wall clock — so no real CUDA context is touched anywhere.

    Class state (`DONE`, the tick counter, the live counter) is process-wide on purpose —
    `tests/run_all.py` runs the whole suite in one process — so every caller runs `reset()`
    (done automatically by `DeviceSpy.__enter__`) rather than relying on whatever an earlier
    test left behind. `_live` is read directly as a plain int by callers that want a
    construction count (`FakeCudaEvent._live` after a block, matching the pre-F6 shape
    `test_pace462_bounded_lookahead.py` read it in) — a class-level int works for this because
    `type(self)._live += 1` rebinds the class attribute, not the instance's."""

    _next_tick = 0
    _live = 0
    DONE = True
    TICK_MS = 250.0   # a synthetic device-ms unit; harmless to share, no test asserts on it

    def __init__(self, blocking=False, enable_timing=False):
        self.blocking = blocking
        self.enable_timing = enable_timing
        self.record_calls = 0
        self.sync_calls = 0
        self._tick = None
        type(self)._live += 1
        self._id = type(self)._live

    def record(self) -> None:
        self.record_calls += 1
        self._tick = type(self)._next_tick
        type(self)._next_tick += 1

    def synchronize(self) -> None:
        self.sync_calls += 1

    def query(self) -> bool:
        return type(self).DONE

    def elapsed_time(self, other) -> float:
        return (other._tick - self._tick) * type(self).TICK_MS

    @classmethod
    def reset(cls) -> None:
        cls._next_tick = 0
        cls._live = 0
        cls.DONE = True


class DeviceSpy:
    """Patches `torch.cuda.device` (the context manager), `is_available`, `current_device`
    and `Event` well enough to drive `tex_runtime/pacing.py` and `tex_runtime/profile.py`'s
    CUDA branches on ANY box, CUDA or not (F6): one shared shape for what
    `test_fixobsroute46_pacing.py`, `test_pace462_bounded_lookahead.py` and
    `test_prof462_device_honest.py` each hand-rolled separately, and had already drifted —
    one dropped the `current=` override, one renamed `calls` to `device_calls`, one skipped
    `current_device` entirely and only survived because the code it drove never called it
    (R1#1 of the v0.46.2 Phase C reuse review; the CI-shape gap it flagged already fired once).

    `current` is the FIXED value `torch.cuda.current_device()` reports (default 0) — fixed
    rather than tracked, so a test can put a genuinely non-current index (e.g. `"cuda:1"`) on
    one side of a comparison and a genuinely current one (`"cuda:0"`/`"cuda"`) on the other.
    `calls` records every device `torch.cuda.device(...)` was actually entered with — empty
    means a device-context-skip fired (already-current device); non-empty means it did not.
    `event_cls` defaults to `FakeCudaEvent` but accepts any drop-in replacement (e.g. a
    subclass whose `record()` raises on demand) without this class needing to know about it.

        with DeviceSpy(current=0) as spy:
            ...
        spy.calls   # -> [] (skipped) or [<device>, ...] (entered)
    """

    def __init__(self, current=0, event_cls=FakeCudaEvent):
        self.calls = []
        self.current = current
        self.event_cls = event_cls
        self._real_available = None
        self._real_device_ctx = None
        self._real_event = None
        self._real_current_device = None

    def __enter__(self) -> "DeviceSpy":
        spy = self
        self._real_available = torch.cuda.is_available
        self._real_device_ctx = torch.cuda.device
        self._real_event = torch.cuda.Event
        self._real_current_device = torch.cuda.current_device

        def _fake_current_device():
            return spy.current

        class _Ctx:
            def __init__(self, dev):
                spy.calls.append(dev)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        torch.cuda.is_available = lambda: True
        torch.cuda.current_device = _fake_current_device
        torch.cuda.device = _Ctx
        torch.cuda.Event = self.event_cls
        self.event_cls.reset()
        return self

    def __exit__(self, *exc) -> bool:
        torch.cuda.is_available = self._real_available
        torch.cuda.current_device = self._real_current_device
        torch.cuda.device = self._real_device_ctx
        torch.cuda.Event = self._real_event
        return False
