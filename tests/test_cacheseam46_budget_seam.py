"""CACHESEAM-46 — the mutation seam for the five budget-tracked stdlib caches.

TRK-187 declined an incrementally-maintained running total for
`tex_memory.enforce_cache_budget` (MEM-1/MEM-4) because the five caches it walks
(`stdlib_core._mip_cache`/`_gauss_mip_cache`/`_grid_buf`/`_sampler_cache`/
`_gauss_kernel_cache`) were mutated by direct dict operations OUTSIDE `tex_memory.py`
(14 production sites in `stdlib_core.py`) plus roughly six test fixtures poking the raw
`OrderedDict`s. This ask closes that: every production mutation now goes through a
`stdlib_core._CacheBudget` instance (one per cache, paired in
`stdlib_core.BUDGET_TRACKED_CACHES`), which keeps a running per-device-type byte total
`enforce_cache_budget` reads in O(1) instead of `_total_cache_bytes`'s O(entries) walk.

This file is the proof, not the mechanism (that lives in `tex_runtime/stdlib_core.py` and
`tex_memory.py`):

  * `test_cacheseam46_drift_free` — a randomized sequence of seam operations across all
    five caches, asserting the running total equals a full recount after EVERY step. This
    is the ask's own bar ("provably drift-free"), not a spot check.
  * `test_cacheseam46_reset_for_test_helper` — the documented escape hatch for a fixture
    that must poke a cache directly: `reset_for_test` recovers an accurate total from a
    raw poke that the seam could not see.
  * `test_cacheseam46_eviction_order_unchanged` — a scripted six-entry sequence proves
    `enforce_cache_budget` still evicts oldest-first and keeps the newest entries, exactly
    as before this ask (no pixel/behaviour change, only where the byte count comes from).
  * `test_cacheseam46_budget_status_query` — the new read-only `cache_budget_status`
    query reports an accurate (limit, usage) pair and never mutates or evicts.

PORTABILITY: pure torch, CPU only (the caches' own eviction logic is already exercised on
CUDA by `test_v018_memory.py`/`test_v015_audit_fixes.py`; this file is about the
bookkeeping, which is device-agnostic). No ComfyUI, no compiler, no numpy.
"""
import os
import random

from helpers import *
from TEX_Wrangle import tex_memory as MEM
from TEX_Wrangle.tex_runtime import stdlib_core as SC


def _entry_for(cache, device="cpu"):
    """A cheap stand-in entry of the right SHAPE for `cache`, mirroring the tuple/tensor
    shapes `stdlib_core.py`'s own builders store (see `BUDGET_TRACKED_CACHES`'s extractors)."""
    if cache is SC._mip_cache or cache is SC._gauss_mip_cache:
        img = torch.zeros(1, 4, 4, 3, device=device)
        return ((1, 4, 4, 3), img, [torch.zeros(1, 2, 2, 3, device=device)])
    if cache is SC._gauss_kernel_cache:
        return (torch.zeros(1, 1, 1, 3, device=device), torch.zeros(1, 1, 3, 1, device=device))
    return torch.zeros(4, 4, device=device)  # _grid_buf / _sampler_cache


def _full_recount(cache, budget, dev_type=None) -> int:
    """The ground truth: the exact arithmetic `tex_memory._total_cache_bytes` uses, over
    ONE cache, via the SAME `budget.extract` the production walk reads."""
    total = 0
    for entry in cache.values():
        if dev_type is None or MEM._entry_dev_type(entry, budget.extract) == dev_type:
            total += MEM._entry_bytes(entry, budget.extract)
    return total


def test_cacheseam46_drift_free(r: SubTestResult):
    print("\n--- CACHESEAM-46: running budget total never drifts from a full recount ---")
    MEM.free_tensor_caches()
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    rng = random.Random(20260925)
    pairs = list(SC.BUDGET_TRACKED_CACHES)
    bad = []
    steps = 0
    try:
        for step in range(400):
            cache, budget = rng.choice(pairs)
            op = rng.choice(("put", "put", "put", "evict", "delete", "touch", "clear"))
            keys = list(cache.keys())
            if op == "put":
                key = ("scripted", rng.randrange(24))
                budget.put(cache, key, _entry_for(cache, rng.choice(devices)))
            elif op == "evict":
                budget.evict_oldest(cache)
            elif op == "delete" and keys:
                budget.delete(cache, rng.choice(keys))
            elif op == "touch" and keys:
                budget.touch(cache, rng.choice(keys))
            elif op == "clear":
                budget.clear(cache)
            # else: no-op this step (e.g. "delete"/"touch" on an empty cache) -- still
            # re-verified below, which is a legitimate (trivial) drift check too.
            steps += 1
            for dev_type in (None,) + tuple(devices):
                got = budget.total(dev_type)
                want = _full_recount(cache, budget, dev_type)
                if got != want:
                    bad.append(f"step {step} ({op}) on {id(cache)}: "
                               f"total(dev_type={dev_type!r})={got} != full recount {want}")
        if bad:
            r.fail("CACHESEAM-46 drift-free",
                   f"{len(bad)} mismatch(es); first 5: " + "; ".join(bad[:5]))
        else:
            r.ok(f"running total matched a full recount after all {steps} scripted operations "
                 f"across {len(pairs)} caches x {1 + len(devices)} dev_type readings")
    except Exception as e:
        r.fail("CACHESEAM-46 drift-free", f"{type(e).__name__}: {e}")
    finally:
        MEM.free_tensor_caches()


def test_cacheseam46_reset_for_test_helper(r: SubTestResult):
    print("\n--- CACHESEAM-46: reset_for_test recovers from a raw dict poke ---")
    MEM.free_tensor_caches()
    try:
        SC._grid_buf_budget.put(SC._grid_buf, ("r", 0), torch.zeros(1, 64, 64, 4))
        # The exact shape the pre-ask test fixtures used (a bare `cache[key] = value`,
        # bypassing the seam on purpose here to prove the recovery path, not to repeat
        # the bug -- production code and the fixed test fixtures never do this anymore).
        SC._grid_buf[("r", 1)] = torch.zeros(1, 64, 64, 4)
        want = 2 * 64 * 64 * 4 * 4
        drifted = SC._grid_buf_budget.total("cpu")
        SC._grid_buf_budget.reset_for_test(SC._grid_buf)
        recovered = SC._grid_buf_budget.total("cpu")
        if drifted == want:
            r.fail("CACHESEAM-46 reset_for_test",
                   "the raw poke did not actually drift the total -- this row proves nothing")
        elif recovered != want:
            r.fail("CACHESEAM-46 reset_for_test",
                   f"reset_for_test left the total wrong (got {recovered}, want {want})")
        else:
            r.ok(f"a raw poke drifted the total ({drifted} != {want}); "
                 f"reset_for_test recovered it exactly")
    except Exception as e:
        r.fail("CACHESEAM-46 reset_for_test", f"{type(e).__name__}: {e}")
    finally:
        MEM.free_tensor_caches()


def test_cacheseam46_eviction_order_unchanged(r: SubTestResult):
    print("\n--- CACHESEAM-46: eviction order is unchanged (oldest-first, newest kept) ---")
    MEM.free_tensor_caches()
    saved = os.environ.get("TEX_CACHE_BUDGET_MB")
    try:
        # Six ~1 MiB entries, oldest (0) to newest (5), through the seam -- the same shape
        # a real cook builds via `_get_grid_buf`, and the same scenario
        # `test_v016_phase2.py::test_m2cpu_and_m1_freeretry` and
        # `test_v015_audit_fixes.py::test_mem1_evict_preserves_graphs` already exercise;
        # this row pins the EXACT surviving set rather than just "fewer than before".
        for i in range(6):
            SC._grid_buf_budget.put(SC._grid_buf, ("scr", i), torch.zeros(1, 256, 256, 4))
        os.environ["TEX_CACHE_BUDGET_MB"] = "2"   # room for exactly two 1 MiB entries
        MEM.enforce_cache_budget("cpu")
        survivors = list(SC._grid_buf.keys())
        want = [("scr", 4), ("scr", 5)]
        if survivors == want:
            r.ok(f"oldest-first eviction unchanged: survivors {survivors} == {want}")
        else:
            r.fail("CACHESEAM-46 eviction order",
                   f"survivors {survivors} != expected {want} (oldest-first, newest-kept)")
    except Exception as e:
        r.fail("CACHESEAM-46 eviction order", f"{type(e).__name__}: {e}")
    finally:
        if saved is None:
            os.environ.pop("TEX_CACHE_BUDGET_MB", None)
        else:
            os.environ["TEX_CACHE_BUDGET_MB"] = saved
        MEM.free_tensor_caches()


def test_cacheseam46_budget_status_query(r: SubTestResult):
    print("\n--- CACHESEAM-46: cache_budget_status is a read-only, accurate query ---")
    MEM.free_tensor_caches()
    saved = os.environ.get("TEX_CACHE_BUDGET_MB")
    try:
        os.environ["TEX_CACHE_BUDGET_MB"] = "5"
        before = MEM.cache_budget_status("cpu")
        assert before["limit_bytes"] == 5 * 1024 * 1024, "limit did not reflect the override"
        assert before["usage_bytes"] == 0, "empty caches must read zero usage"

        entry_bytes = 256 * 256 * 4 * 4
        SC._grid_buf_budget.put(SC._grid_buf, ("q", 0), torch.zeros(1, 256, 256, 4))
        after = MEM.cache_budget_status("cpu")
        assert after["usage_bytes"] == entry_bytes, \
            f"usage did not reflect the put ({after['usage_bytes']} != {entry_bytes})"
        assert after["limit_bytes"] == before["limit_bytes"], "the query must not move the limit"

        # Never mutates or evicts, even called far over budget.
        os.environ["TEX_CACHE_BUDGET_MB"] = "1"
        for i in range(1, 8):
            SC._grid_buf_budget.put(SC._grid_buf, ("q", i), torch.zeros(1, 256, 256, 4))
        n_before = len(SC._grid_buf)
        over_budget = MEM.cache_budget_status("cpu")
        assert len(SC._grid_buf) == n_before, "cache_budget_status evicted entries"
        assert over_budget["usage_bytes"] > over_budget["limit_bytes"], \
            "the scenario should read over budget"
        r.ok("cache_budget_status reports an accurate, read-only (limit, usage) pair")
    except Exception as e:
        r.fail("CACHESEAM-46 budget status query", f"{type(e).__name__}: {e}")
    finally:
        if saved is None:
            os.environ.pop("TEX_CACHE_BUDGET_MB", None)
        else:
            os.environ["TEX_CACHE_BUDGET_MB"] = saved
        MEM.free_tensor_caches()
