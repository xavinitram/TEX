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

FIX-CACHE/K1: `_CacheBudget` above had no lock, so two threads mutating the SAME cache
concurrently (COMPILE-A's warm_call thread vs. a cook thread; the aiohttp
`/tex_wrangle/free_caches` route vs. either) could corrupt the underlying `OrderedDict`'s
internal state — reproduced as a hard access-violation crash pinned at `evict_oldest`'s
`cache.popitem` racing `clear`'s `cache.clear()` (B3 cache-seam finding K1). Every
`_CacheBudget` compound operation (the dict mutation and the running-total update
together) now runs under one `RLock` per cache.

  * `test_cacheseam46_thread_stress` — several cook threads hammering put/evict/touch/
    delete on ONE cache concurrently with a clearer thread, mirroring the two real
    concurrency sources above. Must never crash and must leave the running total exactly
    equal to a full recount (the same drift check `test_cacheseam46_drift_free` runs
    single-threaded, now under real contention). Fixed thread counts and a fixed seed,
    CPU only, a few seconds.
  * `test_cacheseam46_seam_is_load_bearing` — FIX-CACHE/K2: an AST census of
    `stdlib_core.py` proving no code OUTSIDE the `_CacheBudget` class body mutates one of
    the five budget-tracked caches by a subscript store/delete or a `.popitem()`/
    `.clear()`/`.move_to_end()`/`.pop()`/`.setdefault()`/`.update()` call — the seam is
    the only path, and this ratchet catches a future bypass instead of relying on the
    one-time grep the CACHESEAM-46 comment above records.

PORTABILITY: pure torch, CPU only (the caches' own eviction logic is already exercised on
CUDA by `test_v018_memory.py`/`test_v015_audit_fixes.py`; this file is about the
bookkeeping, which is device-agnostic). No ComfyUI, no compiler, no numpy.
"""
import ast
import inspect
import os
import random
import threading

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


def test_cacheseam46_thread_stress(r: SubTestResult):
    """FIX-CACHE/K1 regression, adapted from the B3 audit's scratchpad stress scripts
    (`cacheseam46_thread_stress*.py`). Fixed thread counts and a fixed seed: this is a
    regression test, not a fuzzer, so a red must reproduce deterministically.

    Several "cook" threads hammer put/evict_oldest/touch/delete on ONE budget-tracked
    cache (`_grid_buf`/`_grid_buf_budget`) while a "clearer" thread concurrently calls
    `budget.clear()` -- exactly the aiohttp `/tex_wrangle/free_caches` route racing a
    queued cook. Before the K1 lock this crashed the whole interpreter with an access
    violation inside a handful of iterations (reproduced against this same commit before
    the fix landed); there is no way for a Python-level `try/except` to turn a native
    access violation into a clean assertion failure, so the bar this test actually
    enforces is "the process is still alive to check the total" -- if the lock regresses,
    this test does not fail, it crashes the interpreter, which is a louder signal than
    any assertion could be.
    """
    print("\n--- CACHESEAM-46/K1: concurrent put/evict/touch/delete/clear never drifts "
          "or crashes ---")
    MEM.free_tensor_caches()
    cache, budget = SC._grid_buf, SC._grid_buf_budget
    N_COOK_THREADS = 6
    OPS_PER_THREAD = 1500
    KEY_SPACE = 40
    CLEAR_INTERVAL = 0.0005

    errors = []
    errors_lock = threading.Lock()
    stop_flag = threading.Event()

    def record_error(name, exc):
        with errors_lock:
            errors.append((name, type(exc).__name__, str(exc)))

    def cook_worker(idx):
        rng = random.Random(1000 + idx)  # deterministic per-thread op sequence
        name = f"cook-{idx}"
        for _ in range(OPS_PER_THREAD):
            key = ("scr", rng.randrange(KEY_SPACE))
            try:
                op = rng.choice(("put", "put", "put", "get_touch", "evict", "delete"))
                if op == "put":
                    budget.put(cache, key, torch.empty(4, 8, 8, 2, dtype=torch.float32))
                    if len(cache) > 16:
                        budget.evict_oldest(cache)
                elif op == "get_touch":
                    # the exact get-then-touch TOCTOU shape _get_grid_buf-style builders
                    # use: `cache.get` runs OUTSIDE the lock, so the key can legitimately
                    # be gone by the time `touch` runs -- K1 made `touch` a no-op then,
                    # not a KeyError.
                    if cache.get(key) is not None:
                        budget.touch(cache, key)
                elif op == "evict":
                    budget.evict_oldest(cache)
                elif op == "delete":
                    budget.delete(cache, key)  # K1: a no-op if already gone
            except Exception as e:  # noqa: BLE001 -- any exception here is the failure
                record_error(name, e)

    def host_clearer():
        while not stop_flag.is_set():
            try:
                budget.clear(cache)
            except Exception as e:  # noqa: BLE001
                record_error("host-clearer", e)
            time.sleep(CLEAR_INTERVAL)

    try:
        threads = [threading.Thread(target=cook_worker, args=(i,), name=f"cook-{i}")
                   for i in range(N_COOK_THREADS)]
        clearer = threading.Thread(target=host_clearer, name="host-clearer", daemon=True)
        clearer.start()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        stop_flag.set()
        clearer.join(timeout=2)

        if any(t.is_alive() for t in threads):
            r.fail("CACHESEAM-46/K1 thread stress", "a cook thread did not finish (deadlock?)")
            return
        if errors:
            by_type = {}
            for _name, etype, _msg in errors:
                by_type[etype] = by_type.get(etype, 0) + 1
            r.fail("CACHESEAM-46/K1 thread stress",
                   f"{len(errors)} exception(s) raised inside the seam: {by_type}; "
                   f"first: {errors[0]}")
            return

        got = budget.total(None)
        want = 0
        for entry in cache.values():
            for t in budget.extract(entry):
                if isinstance(t, torch.Tensor):
                    want += t.untyped_storage().nbytes()
        cache_keys = set(cache.keys())
        tracked_keys = set(budget._per_key.keys())
        if got != want:
            r.fail("CACHESEAM-46/K1 thread stress",
                   f"post-run drift: running total={got} != full recount={want}")
        elif cache_keys != tracked_keys:
            r.fail("CACHESEAM-46/K1 thread stress",
                   f"bookkeeping/dict key-set mismatch: "
                   f"{len(cache_keys - tracked_keys)} untracked, "
                   f"{len(tracked_keys - cache_keys)} tracked-but-absent")
        else:
            r.ok(f"{N_COOK_THREADS} cook threads x {OPS_PER_THREAD} ops, concurrent with a "
                 f"clearer thread: no crash, no exception, total()={got} matches a full "
                 f"recount, {len(cache_keys)} key(s) tracked exactly")
    finally:
        stop_flag.set()
        MEM.free_tensor_caches()


def test_cacheseam46_seam_is_load_bearing(r: SubTestResult):
    """FIX-CACHE/K2: the seam is load-bearing -- automate the CACHESEAM-46 comment's own
    claim ("grep finds no other `[key] =`, `.popitem(`, `.clear()` or `.move_to_end(`
    against these five names in this file") as a ratchet instead of a one-time manual
    grep, so a future direct dict mutation of a budget-tracked cache is CAUGHT here
    rather than merely absent today.

    An AST census of `stdlib_core.py`: every reference to one of the five budget-tracked
    cache names that is a subscript store/delete (`cache[key] = ...` / `del cache[key]`)
    or a mutating-method call (`.popitem(`/`.clear(`/`.move_to_end(`/`.pop(`/
    `.setdefault(`/`.update(`) must fall inside the `_CacheBudget` class body's own line
    span -- that is the one place `BUDGET_TRACKED_CACHES`' five instances live and the
    only intended mutation path (K1's lock lives there too, so a bypass would also be an
    unlocked mutation).

    K2 was optional if K1's lock design already made a bypass safe; this row is the cheap
    half of it -- catching a bypass in review is strictly better than merely surviving one
    at runtime, and it costs one AST walk of one file. The caches themselves stay plain
    `OrderedDict`s (DOC-7d's census in `test_v018_docs.py` keys off that declaration
    shape), so this is a second, narrower census beside DOC-7d's, not a replacement --
    DOC-7d asks "is every module-level store documented", this asks "does anything
    outside the seam touch these five specifically"."""
    print("\n--- CACHESEAM-46/K2: no direct mutation of a budget-tracked cache outside "
          "_CacheBudget ---")
    try:
        source_path = inspect.getsourcefile(SC)
        with open(source_path, encoding="utf-8") as fh:
            source = fh.read()
        tree = ast.parse(source, filename=source_path)

        # The five names, read from the module itself (never hand-duplicated) by matching
        # each cache object in BUDGET_TRACKED_CACHES back to its module-level attribute name.
        by_id = {id(cache): attr for attr, cache in vars(SC).items()
                 if any(cache is c for c, _b in SC.BUDGET_TRACKED_CACHES)}
        cache_names = {by_id[id(c)] for c, _b in SC.BUDGET_TRACKED_CACHES if id(c) in by_id}
        if len(cache_names) != len(SC.BUDGET_TRACKED_CACHES):
            r.fail("CACHESEAM-46/K2 seam census",
                   f"could not resolve all {len(SC.BUDGET_TRACKED_CACHES)} cache names by "
                   f"identity (got {sorted(cache_names)}) -- fix the derivation")
            return

        span = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "_CacheBudget":
                span = (node.lineno, node.end_lineno)
                break
        if span is None:
            r.fail("CACHESEAM-46/K2 seam census", "_CacheBudget class not found by AST walk")
            return
        lo, hi = span

        parent = {}
        for n in ast.walk(tree):
            for c in ast.iter_child_nodes(n):
                parent[c] = n

        mutating_methods = {"popitem", "clear", "move_to_end", "pop", "setdefault", "update"}
        bypasses = []
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Name) and node.id in cache_names):
                continue
            if lo <= node.lineno <= hi:
                continue  # inside the seam class itself
            p = parent.get(node)
            if isinstance(p, ast.Subscript) and p.value is node and isinstance(p.ctx, (ast.Store, ast.Del)):
                bypasses.append(f"{source_path}:{node.lineno}: {node.id}[...] store/delete")
                continue
            if isinstance(p, ast.Attribute) and p.value is node and p.attr in mutating_methods:
                gp = parent.get(p)
                if isinstance(gp, ast.Call) and gp.func is p:
                    bypasses.append(f"{source_path}:{node.lineno}: {node.id}.{p.attr}(...)")

        if bypasses:
            r.fail("CACHESEAM-46/K2 seam bypass",
                   f"{len(bypasses)} direct mutation(s) of a budget-tracked cache outside "
                   f"_CacheBudget: " + "; ".join(bypasses))
        else:
            r.ok(f"no direct mutation of any of {sorted(cache_names)} outside the "
                 f"_CacheBudget class body (lines {lo}-{hi})")
    except Exception as e:
        r.fail("CACHESEAM-46/K2 seam census", f"{type(e).__name__}: {e}")
