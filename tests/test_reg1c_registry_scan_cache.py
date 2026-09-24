"""REG-1c — the registry-derived non-spatial-argument lookup becomes O(1) after first use.

`stdlib_registry.non_spatial_args_by_name()` used to rebuild its `{name: positions}` view by
scanning the WHOLE builtin registry on every single call.
`interpreter._collect_binding_reads_and_non_spatial` calls it once per AST walk, and that walk
is itself memoized per `Program` OBJECT (`interpreter._READS_MEMO`) — but a cold compile builds
a fresh `Program` every time, so the per-program memo never hits and the registry-wide scan
re-paid on every single cold compile, not just once.

This file proves the fix by COUNT, not time, per the house rule (a shared box makes any timing
assertion here worthless — see `tests/test_bench2_counts.py`'s own docstring for the measured
null-control spread): the scan now happens at most once per process, and exactly once more per
new `@stdlib` registration, never once per compile.
"""
from helpers import *
from TEX_Wrangle.tex_runtime import stdlib_registry as R


class _CountingRegistry(list):
    """A drop-in stand-in for `stdlib_registry.REGISTRY` that counts how many times it is
    iterated — the direct measure of how many times `non_spatial_args_by_name()` rebuilds its
    cache, with no timing involved."""

    def __init__(self, *a):
        super().__init__(*a)
        self.scans = 0

    def __iter__(self):
        self.scans += 1
        return super().__iter__()


def test_reg1c_scan_count_after_first_use(r: SubTestResult):
    print("\n--- REG-1c: non_spatial_args_by_name() scans the registry at most once ---")
    saved_registry = R.REGISTRY
    saved_cache = dict(R._NON_SPATIAL_CACHE)
    saved_ready = R._NON_SPATIAL_CACHE_READY
    try:
        counting = _CountingRegistry(saved_registry)
        R.REGISTRY = counting
        R._NON_SPATIAL_CACHE_READY = False   # simulate a cold process: cache not yet built

        first = R.non_spatial_args_by_name()
        for _ in range(9):
            R.non_spatial_args_by_name()

        if counting.scans != 1:
            r.fail("REG-1c cold-compile scan count",
                   f"expected exactly 1 registry scan across 10 calls (build-once, reuse "
                   f"after), got {counting.scans} -- the whole-registry rebuild is re-paying "
                   f"per call again")
        else:
            r.ok("registry scanned exactly once across 10 calls (was: once PER call -- 10 -- "
                 "before REG-1c)")

        if first is not R.non_spatial_args_by_name():
            r.fail("REG-1c cache identity", "repeated calls returned different dict objects "
                   "-- the cache is being rebuilt even though REGISTRY hasn't changed")
        else:
            r.ok("repeated calls return the SAME cached dict object")
    finally:
        R.REGISTRY = saved_registry
        R._NON_SPATIAL_CACHE.clear()
        R._NON_SPATIAL_CACHE.update(saved_cache)
        R._NON_SPATIAL_CACHE_READY = saved_ready


def test_reg1c_late_registration_still_seen(r: SubTestResult):
    print("\n--- REG-1c: a builtin registered AFTER the first lookup still invalidates the cache ---")
    saved_registry_snapshot = list(R.REGISTRY)
    saved_cache = dict(R._NON_SPATIAL_CACHE)
    saved_ready = R._NON_SPATIAL_CACHE_READY
    name = "__test_reg1c_late_fn__"
    try:
        # Warm the cache BEFORE the new builtin exists -- the shape that would silently break
        # if invalidation were missing (a first-use cache that is never rebuilt).
        before = R.non_spatial_args_by_name()
        if name in before:
            r.fail("REG-1c late-registration setup",
                   f"{name!r} is already registered -- pick a name that cannot collide")
            return

        @R.stdlib(name, non_spatial_args=(0,))
        def _dummy(*_a):
            return None

        after = R.non_spatial_args_by_name()
        if after.get(name) != (0,):
            r.fail("REG-1c late registration",
                   f"a builtin registered after the first lookup was not seen -- "
                   f"non_spatial_args_by_name() returned {after.get(name)!r} for {name!r}, "
                   f"expected (0,); the cache is stale after a new @stdlib registration")
        else:
            r.ok(f"a builtin registered AFTER the first lookup ({name!r}) is still seen -- "
                 f"the cache invalidates correctly on registration")
    finally:
        R.REGISTRY[:] = saved_registry_snapshot
        R._NON_SPATIAL_CACHE.clear()
        R._NON_SPATIAL_CACHE.update(saved_cache)
        R._NON_SPATIAL_CACHE_READY = saved_ready
