"""OVH-47 (TRK-211) — one memoized device parse instead of up to three per cook.

WHAT THE LANE CHANGED. `tex_memory.py` had seven call sites (`cache_budget_bytes`,
`enforce_cache_budget`, `governor_budget`, `free_memory_hint`, `device_total_mem`,
`trim_reserved_pool`, and `cache_budget_status`) each independently re-parsing the SAME
raw `device` value (`tex_engine`'s `ctx.device` -- a plain `str` on the default path) into
a fresh `torch.device` via `torch.device(device) if not isinstance(device, torch.device)
else device`. `run()`'s own per-cook body alone reaches three of them
(`enforce_cache_budget` -> `cache_budget_bytes` internally, then `trim_reserved_pool`),
so a single default-path cook paid for the identical parse three times. This is a
per-function attribution finding from OVH-47's pin(999b2f9)->head cProfile tottime
comparison (`cache_budget_bytes`/`enforce_cache_budget`/`trim_reserved_pool` all showed
small but consistent per-call tottime growth with an UNCHANGED call count -- the
"invisible to a call-count diff" bucket TRK-211 asks this lane to find); the redundant
re-parse was already present at the pin, so this closes a pre-existing inefficiency
rather than a new regression, and is reported honestly as such here.

THE FIX: `_as_device(device)`, memoized by the raw `device` value in `_device_obj_cache`
(mirroring the existing `_total_mem_cache` pattern just below it in this file) -- bounded
by the number of DISTINCT device values a process ever sees (a handful, never per-cook),
and never invalidated (a device string always resolves to the same `torch.device`).
Byte-identical: `_as_device` returns exactly what the inlined ternary used to construct,
just once per distinct value instead of once per call.

RED-FIRST: `test_ovh47_device_parse_memoized_across_calls` counts real `torch.device(...)`
constructor calls (not `isinstance` checks) across a run()-shaped sequence --
`enforce_cache_budget` then `trim_reserved_pool` then `cache_budget_status`, three
functions, same device string -- and asserts exactly ONE construction total. At the base
sha (before this lane's memo) this reads three. `test_ovh47_device_parse_distinct_values`
proves the memo is keyed by VALUE, not a single global answer: two different device
strings each still resolve correctly and independently. `test_ovh47_device_object_passthrough`
proves an already-a-`torch.device` argument short-circuits without ever reaching the
constructor (the `isinstance` branch).

PORTABILITY: every assertion here counts calls with a "cpu"/"cuda:0" STRING device --
`torch.device("cpu")` and `torch.device("cuda:0")` both construct with no CUDA runtime
required (`_as_device` never queries the driver; only unwraps/parses the string), so this
runs identically with or without a CUDA-capable box."""
from helpers import *

from contextlib import contextmanager

from TEX_Wrangle import tex_memory

_RealDevice = torch.device


class _CountingDeviceMeta(type):
    """`isinstance(x, _CountingDevice)` must answer exactly as `isinstance(x, torch.device)`
    would -- `_as_device`'s own first line depends on that -- so this metaclass delegates the
    check to the REAL type. `torch.device` is an immutable C type (no `__init__`/subclass
    patch point), so counting constructions means substituting the NAME `torch.device` points
    at, not modifying the real class -- and the substitute must stay isinstance-transparent."""

    def __instancecheck__(cls, instance):
        return isinstance(instance, _RealDevice)


@contextmanager
def _fresh_device_cache():
    """Each test gets an empty memo -- a prior test's cached values must not answer for
    THIS test's own call-counting, and must not leak into a later test either."""
    saved = dict(tex_memory._device_obj_cache)
    tex_memory._device_obj_cache.clear()
    try:
        yield
    finally:
        tex_memory._device_obj_cache.clear()
        tex_memory._device_obj_cache.update(saved)


@contextmanager
def _counting_torch_device():
    """Substitute `torch.device` with an isinstance-transparent counting stand-in for the
    duration of the block -- the counter IS the assertion, the same shape
    `test_perf6_free_memory_once.py`'s `_FakeHost.calls` counts `host.get_free_memory`."""
    calls = {"n": 0}

    class _CountingDevice(metaclass=_CountingDeviceMeta):
        def __new__(cls, *a, **kw):
            calls["n"] += 1
            return _RealDevice(*a, **kw)

    saved = torch.device
    torch.device = _CountingDevice
    try:
        yield calls
    finally:
        torch.device = saved


def test_ovh47_device_parse_memoized_across_calls(r: SubTestResult):
    print("\n--- one cook's worth of tex_memory calls parses its device ONCE, not 3x ---")
    with _fresh_device_cache():
        with _counting_torch_device() as calls:
            tex_memory.enforce_cache_budget("cpu")
            tex_memory.trim_reserved_pool("cpu", 0)
            tex_memory.cache_budget_status("cpu")
        if calls["n"] == 1:
            r.ok("enforce_cache_budget + trim_reserved_pool + cache_budget_status on the "
                 "same device string -> 1 torch.device() construction")
        else:
            r.fail("OVH-47 device parse memo",
                   f"expected exactly 1 torch.device('cpu') construction across three "
                   f"tex_memory calls on the same raw device string, got {calls['n']} "
                   f"(pre-memo base sha reads 3 -- one per call site)")


def test_ovh47_device_parse_distinct_values(r: SubTestResult):
    print("\n--- the memo is keyed by VALUE: two device strings resolve independently ---")
    with _fresh_device_cache():
        with _counting_torch_device() as calls:
            a1 = tex_memory._as_device("cpu")
            b1 = tex_memory._as_device("cuda:0")
            a2 = tex_memory._as_device("cpu")
            b2 = tex_memory._as_device("cuda:0")
        if calls["n"] == 2:
            r.ok("two distinct device strings -> exactly 2 constructions total (one per "
                 "distinct value), repeats are free")
        else:
            r.fail("OVH-47 device parse memo (distinct values)",
                   f"expected exactly 2 constructions, got {calls['n']}")
        if a1 is a2 and a1.type == "cpu":
            r.ok("'cpu' memo hit returns the SAME object on repeat")
        else:
            r.fail("OVH-47 device parse memo (cpu identity)", "second 'cpu' call did not "
                   "return the cached object")
        if b1 is b2 and b1.type == "cuda":
            r.ok("'cuda:0' memo hit returns the SAME object on repeat")
        else:
            r.fail("OVH-47 device parse memo (cuda identity)", "second 'cuda:0' call did "
                   "not return the cached object")
        if a1 is not b1:
            r.ok("'cpu' and 'cuda:0' never share a cached object")
        else:
            r.fail("OVH-47 device parse memo (cross-value)", "two distinct device strings "
                   "collided on one cached object")


def test_ovh47_device_object_passthrough(r: SubTestResult):
    print("\n--- an already-torch.device argument never reaches the constructor ---")
    with _fresh_device_cache():
        pre_built = torch.device("cpu")
        with _counting_torch_device() as calls:
            out = tex_memory._as_device(pre_built)
        if calls["n"] == 0:
            r.ok("passing a torch.device short-circuits on isinstance, 0 constructions")
        else:
            r.fail("OVH-47 device parse memo (passthrough)",
                   f"a torch.device argument must not reach the constructor, saw "
                   f"{calls['n']} constructions")
        if out is pre_built:
            r.ok("the passthrough returns the SAME object, not a copy")
        else:
            r.fail("OVH-47 device parse memo (passthrough identity)",
                   "_as_device returned a different object for an already-torch.device arg")
