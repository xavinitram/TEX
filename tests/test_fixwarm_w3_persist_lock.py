"""
FIX-WARM W3 (v0.48 Phase C, B3#3) -- `warm_state.persist(force=True)` gains its first
same-process concurrent caller (`prewarm()`, reachable via `tex_api.prewarm_async`), and its
ordering invariant (snapshot BEFORE `drop_prefix`, ENG-13) was enforced only by there having
been exactly one caller at a time -- incidentally, not by a lock. Before AUTO-48, every
`persist()` call in one process ran on whatever single thread called `prewarm()` or a cook;
`prewarm_async` is the first path that puts a real, concurrent, `force=True` caller into the
same process as the cook thread's own `note_update()`/`persist(force=False)` calls.

Red at 5ae6288: two threads calling `persist(force=True)` concurrently, with
`Journal.drop_prefix` slowed to widen the window, race on the module globals
`_last_persist`/`_path_cache`/`_tag_cache` and on the snapshot-then-`drop_prefix` sequence
with no serialization at all -- this test pins that a lock now makes the two calls run
ONE AT A TIME (never interleaved), rather than merely hoping the race stays benign.
"""
import threading
import time

from helpers import *  # noqa: F401,F403  (cold_engine_state)
from TEX_Wrangle.tex_runtime import warm_state


def test_fixwarm_w3_persist_serializes_concurrent_callers(r: SubTestResult):
    print("\n--- FIX-WARM W3: warm_state.persist(force=True) serializes concurrent callers ---")
    with cold_engine_state():
        warm_state.ensure_loaded()
        from TEX_Wrangle.tex_runtime import graphed
        graphed._capturable_memo["fp_w3_test"] = (True, 3)

        active = {"n": 0}
        max_active = {"n": 0}
        active_lock = threading.Lock()

        import TEX_Wrangle.tex_recovery as tex_recovery
        orig_atomic_write_json = tex_recovery.atomic_write_json

        def slow_atomic_write_json(*a, **kw):
            with active_lock:
                active["n"] += 1
                max_active["n"] = max(max_active["n"], active["n"])
            try:
                time.sleep(0.1)   # widen the window so two real callers can overlap
                return orig_atomic_write_json(*a, **kw)
            finally:
                with active_lock:
                    active["n"] -= 1

        # `persist()` looks up `atomic_write_json` via a deferred `from ..tex_recovery import
        # atomic_write_json` inside the function body -- patch the module attribute it
        # resolves against.
        tex_recovery.atomic_write_json = slow_atomic_write_json
        errors = []

        def worker():
            try:
                warm_state.persist(force=True)
            except Exception as e:  # pragma: no cover
                errors.append(e)

        try:
            t0 = threading.Thread(target=worker)
            t1 = threading.Thread(target=worker)
            t0.start()
            t1.start()
            t0.join(timeout=10)
            t1.join(timeout=10)
        finally:
            tex_recovery.atomic_write_json = orig_atomic_write_json

        try:
            assert not errors, f"worker thread(s) raised: {errors}"
            assert max_active["n"] == 1, (
                f"two concurrent persist(force=True) callers overlapped inside the "
                f"write ({max_active['n']} concurrently active) -- persist() is not "
                f"serialized against itself")
            r.ok(f"max concurrently-active persist writers: {max_active['n']}")
        except AssertionError as e:
            r.fail("FIX-WARM W3 persist serialization", str(e))
