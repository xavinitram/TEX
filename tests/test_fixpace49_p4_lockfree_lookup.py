"""FIX-PACE49 P4 -- `_pace49_cost_gate` runs `_cost_lookup` (which
acquires `_COST_LOCK`) on *every* poll that reaches the already-economized "device caught
up" skip path once a call site is warm. This is exactly the skip path PACE-47b's own P3 fix
(v0.47) measured and removed a lock/eager-resolution cost from ("~50-75% of the paced skip
path's own per-poll regression"); PACE-49 put a lock back onto it. Measured:
a locked dict lookup costs 105.2 ns/call vs. 43.0 ns/call unlocked
(+145%) on this box.

The fix: `_cost_lookup` becomes lock-free by design -- a dict `.get()` plus two list-index
reads is safe to run WITHOUT `_COST_LOCK` under the GIL (no single Python attribute/item
read can observe a torn write; the worst a concurrent writer can hand this read is last
cook's `(ewma_ms, samples)` OR the freshest one, never a mix of unrelated fields), the same
safety argument `paced_check`'s own `last_confirmed_done` identity cache already relies on
for ITS lock-free read. Only `_cost_feed` (the write side, still guarding `_COST_TABLE`'s
own structural mutations -- `popitem`/insertion/`move_to_end`) keeps the lock.

Proven here by holding `_COST_LOCK` on another thread and confirming `_cost_lookup` still
returns promptly on the calling thread -- a real would-be deadlock/stall probe, not a timing
assertion (no `@pytest.mark.timing` needed; this is a counts/behaviour test that would HANG
were it not for a bounded `join(timeout=...)`, matching this file's own no-real-CUDA-needed
style). RED at base `32f6917` (after P1-P3 land, `_cost_lookup` still holds `_COST_LOCK`):
the lookup thread does not finish before the lock-holder releases.
"""
import threading

from TEX_Wrangle.tex_runtime import pacing as _pace


def test_cost_lookup_never_blocks_on_a_concurrently_held_cost_lock(r):
    print("\n--- FIX-PACE49 P4: _cost_lookup must not block behind a held _COST_LOCK ---")
    key = ("hot-site", 0, 3)
    anchor = object()
    _pace._cost_feed(key, 5.0, anchor)   # a real, warm-shaped entry to look up

    holder_acquired = threading.Event()
    release_now = threading.Event()

    def _hold_lock():
        _pace._COST_LOCK.acquire()
        holder_acquired.set()
        release_now.wait(timeout=5.0)
        _pace._COST_LOCK.release()

    holder = threading.Thread(target=_hold_lock)
    holder.start()
    got_lock = holder_acquired.wait(timeout=5.0)

    result = {}

    def _do_lookup():
        result["value"] = _pace._cost_lookup(key, anchor)
        result["done"] = True

    looker = threading.Thread(target=_do_lookup)
    looker.start()
    # A lock-free read finishes in microseconds; 0.2s is generous slack for a shared,
    # possibly-busy box while still being far shorter than the lock-holder's own hold time
    # below, so a genuinely BLOCKED lookup reliably fails this join.
    looker.join(timeout=0.2)
    finished_promptly = not looker.is_alive()

    release_now.set()
    looker.join(timeout=5.0)
    holder.join(timeout=5.0)

    if not got_lock:
        r.fail("FIX-PACE49 P4 setup", "the lock-holder thread never acquired _COST_LOCK")
    elif finished_promptly and result.get("value") == (5.0, 1):
        r.ok(f"_cost_lookup returned {result['value']} promptly while _COST_LOCK was held "
             f"by another thread -- lock-free, as designed")
    else:
        r.fail("FIX-PACE49 P4 lock-free lookup",
               f"finished_promptly={finished_promptly}, result={result} -- _cost_lookup "
               f"appears to still block behind a concurrently held _COST_LOCK")


def test_cost_feed_still_holds_the_lock_for_its_own_structural_mutations(r):
    """The write side is NOT this fix's target -- `_cost_feed` must still serialize against
    another concurrent `_cost_feed` (the OrderedDict structural-mutation hazard the lock
    exists for in the first place). Same probe, mirrored: a feed on another thread holding
    the lock must still make a SECOND feed on this thread wait for it."""
    print("\n--- FIX-PACE49 P4: _cost_feed (the write side) still serializes on "
          "_COST_LOCK ---")
    key = ("hot-site-2", 0, 3)
    anchor = object()

    holder_acquired = threading.Event()
    release_now = threading.Event()

    def _hold_lock():
        _pace._COST_LOCK.acquire()
        holder_acquired.set()
        release_now.wait(timeout=5.0)
        _pace._COST_LOCK.release()

    holder = threading.Thread(target=_hold_lock)
    holder.start()
    got_lock = holder_acquired.wait(timeout=5.0)

    feeder = threading.Thread(target=_pace._cost_feed, args=(key, 1.0, anchor))
    feeder.start()
    feeder.join(timeout=0.2)
    blocked_as_expected = feeder.is_alive()

    release_now.set()
    feeder.join(timeout=5.0)
    holder.join(timeout=5.0)

    if not got_lock:
        r.fail("FIX-PACE49 P4 write-side setup", "lock-holder never acquired _COST_LOCK")
    elif blocked_as_expected:
        r.ok("_cost_feed correctly waited for the concurrently held _COST_LOCK before "
             "mutating _COST_TABLE")
    else:
        r.fail("FIX-PACE49 P4 write-side regression",
               "_cost_feed did not wait for a concurrently held _COST_LOCK -- the write "
               "side's own structural-mutation guard appears to have been removed too")
