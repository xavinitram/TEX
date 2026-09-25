"""v0.45 SPILL-45 — `ResultCache.spill(key) -> bool`.

An embedding host's ask: a way to spill ONE named entry to the disk tier on demand, out of
`evict_bytes`'s own oldest-first order. The reason it's needed at all —
`evict_bytes` never drops below one resident entry, so the newest `put` (or the sole entry
in a small cache) is the one frame it can structurally never reach. `spill` is mechanism,
not policy: the caller names the key, and this moves exactly that one; nothing about victim
CHOICE changes anywhere else.

Four rows:
  * the newest put (evict_bytes-unreachable) spills and round-trips bit-exact via `get`;
  * absent and already-spilled both answer `False` — the same check (`key` not RAM-resident),
    since spilling removes an entry from `_ram`;
  * no reachable disk tier answers `False` and — the case a lesser implementation could get
    wrong — leaves the entry sitting safely in RAM rather than losing it;
  * a threaded race against ordinary `get`/`put` traffic on the same table, checked once
    everything stops (the ratchet's own idiom: exact answers after the storm, not during it).

CPU-only: `spill` touches no device-placement logic (`_remove`/`_pending_spills`/`_spill`
already run identically off `evict_bytes`, which the CUDA rows of `test_v033_cache8.py`
already cover for the residency ladder), so this file adds nothing by looping CUDA too.
"""
import os
import tempfile
import threading
import time

import torch

from helpers import make_gradient_frame as _frame
from TEX_Wrangle import tex_results


def _recount(c):
    """The per-device byte buckets recomputed from the live entries, vs. as maintained —
    the same cross-check `test_v033_cache8.py`'s race test uses to catch a lost/duplicated
    accounting update under contention."""
    with c._lock:
        recount = {"cuda": 0, "cpu": 0}
        for e in c._ram.values():
            recount[tex_results._dev_bucket(e.device)] += e.nbytes
        return recount, dict(c._bytes_by_dev)


def test_spill45_reaches_the_newest_put_and_round_trips_bit_exact(r):
    print("\n--- SPILL-45: spill() moves the evict_bytes-unreachable newest entry ---")
    try:
        with tempfile.TemporaryDirectory() as d:
            c = tex_results.ResultCache(cache_dir=d)
            frame = _frame(res=48, scale=1.7)
            c.put("newest", frame)
            # `evict_bytes` never drops below one entry, so the sole (= newest) entry
            # survives even a huge ask — this is the unreachable case `spill` exists for.
            freed = c.evict_bytes(10 << 20)
            assert freed == 0 and "newest" in c, \
                "evict_bytes must not have touched the sole entry (test premise broken)"
            st0 = c.stats()

            ok = c.spill("newest")
            assert ok is True, "spill() of a RAM-resident key must return True"
            assert "newest" not in c, "spill() must remove the entry from the RAM tier"

            st1 = c.stats()
            assert st1["spills"] == st0["spills"] + 1, "spills counter did not advance"
            assert st1["evictions"] == st0["evictions"], \
                "spill() is host-directed, not budget pressure — evictions must not move"

            restored = c.get("newest")
            assert restored is not None, "the spilled frame could not be restored"
            assert torch.equal(restored.cpu(), frame.cpu()), \
                "spill -> get round-trip was not bit-exact"
            r.ok(f"newest-put spilled ({st1['spills']} spills) and restored bit-exact")
    except Exception as e:
        r.fail("SPILL-45 round-trip", f"{type(e).__name__}: {e}")


def test_spill45_absent_and_already_spilled_are_both_false(r):
    print("\n--- SPILL-45: absent and already-on-disk both answer False ---")
    try:
        with tempfile.TemporaryDirectory() as d:
            c = tex_results.ResultCache(cache_dir=d)

            never_put = c.spill("never-put-this-key")
            assert never_put is False, "an absent key must answer False"

            frame = _frame(res=32)
            c.put("k", frame)
            first = c.spill("k")
            assert first is True, "the first spill of a resident key must succeed"

            again = c.spill("k")
            assert again is False, \
                "a key spill() already moved out of RAM must answer False, not re-spill"

            # Restoring it back to RAM and spilling it again is a SEPARATE, later act — not
            # what this row is about — but confirms the frame was never lost by the no-op.
            got = c.get("k")
            assert got is not None and torch.equal(got.cpu(), frame.cpu())
            r.ok("absent -> False; already-spilled -> False; the frame survives either way")
    except Exception as e:
        r.fail("SPILL-45 absent/already-spilled", f"{type(e).__name__}: {e}")


def test_spill45_no_disk_tier_leaves_the_frame_in_ram(r):
    print("\n--- SPILL-45: an unreachable disk tier answers False and loses nothing ---")
    try:
        with tempfile.TemporaryDirectory() as d:
            # `_spill_dir` joins `cache_dir` with "results" and makedirs it; pointing
            # `cache_dir` at a plain FILE makes that makedirs fail (NotADirectoryError /
            # FileNotFoundError, both OSError) on every platform — a portable, deterministic
            # stand-in for "no disk tier configured" that needs no permission trickery.
            blocker = os.path.join(d, "blocker")
            with open(blocker, "w") as f:
                f.write("x")
            c = tex_results.ResultCache(cache_dir=blocker)
            frame = _frame(res=32)
            c.put("k", frame)

            ok = c.spill("k")
            assert ok is False, "an unreachable disk tier must answer False"
            assert "k" in c, \
                "a spill() that cannot reach disk must not remove the entry from RAM"
            got = c.get("k")
            assert got is not None and torch.equal(got.cpu(), frame.cpu()), \
                "the frame must still be servable from RAM after a failed spill"
            r.ok("no disk tier: False, and the frame stayed resident and servable")
    except Exception as e:
        r.fail("SPILL-45 no disk tier", f"{type(e).__name__}: {e}")


def test_spill45_races_get_and_put(r):
    """`spill` reuses `evict_bytes`'s own `_remove`/`_pending_spills`/`_claim_spill_ticket`/
    `_drain_spills` path, which is shared, locked, multi-writer state — so it is raced against
    ordinary `get`/`put` traffic on the SAME keys exactly as `test_v033_cache8.py`'s own race
    test does for `touch`/`in`. Checked once every thread has stopped: nothing raised or hung,
    every hit was bit-exact for its own key, the per-device byte buckets agree with a fresh
    recount, and the drain queue is empty (nothing left half-written)."""
    print("\n--- SPILL-45: spill() raced against get/put ---")
    try:
        with tempfile.TemporaryDirectory() as d:
            c = tex_results.ResultCache(cache_dir=d, budget_mb=64)
            keys = [f"k{i}" for i in range(8)]
            frames = [_frame(res=24, scale=1.0 + i * 0.01) for i in range(8)]
            for k, f in zip(keys, frames):
                c.put(k, f)
            stop, errors = threading.Event(), []
            n = {"put": 0, "get": 0, "spill": 0, "wrong": 0}

            def writer():
                i = 0
                try:
                    while not stop.is_set():
                        j = i % len(keys)
                        c.put(keys[j], frames[j])
                        n["put"] += 1
                        i += 1
                except Exception as exc:          # noqa: BLE001 — that IS the finding
                    errors.append(f"writer: {type(exc).__name__}: {exc}")

            def reader():
                i = 0
                try:
                    while not stop.is_set():
                        j = (5 * i) % len(keys)
                        got = c.get(keys[j])
                        n["get"] += 1
                        if got is not None and not torch.equal(got, frames[j]):
                            n["wrong"] += 1
                        i += 1
                except Exception as exc:          # noqa: BLE001
                    errors.append(f"reader: {type(exc).__name__}: {exc}")

            def spiller():
                i = 0
                try:
                    while not stop.is_set():
                        c.spill(keys[(3 * i) % len(keys)])
                        n["spill"] += 1
                        i += 1
                except Exception as exc:          # noqa: BLE001
                    errors.append(f"spiller: {type(exc).__name__}: {exc}")

            threads = [threading.Thread(target=writer, daemon=True),
                       threading.Thread(target=reader, daemon=True),
                       threading.Thread(target=spiller, daemon=True)]
            for t in threads:
                t.start()
            time.sleep(0.5)
            stop.set()
            for t in threads:
                t.join(timeout=30.0)
            hung = [t.name for t in threads if t.is_alive()]

            recount, buckets = _recount(c)
            with c._lock:
                idle = len(c._pending_spills)

            progressed = all(n[k] > 0 for k in ("put", "get", "spill"))
            ok = (not errors and not hung and progressed and n["wrong"] == 0
                  and recount == buckets and idle == 0)
            r.ok(f"spill/get/put raced: {n['spill']} spills, {n['get']} gets, "
                 f"{n['put']} puts, no corruption") if ok else \
                r.fail("SPILL-45 race",
                       f"errors={errors} hung={hung} progressed={progressed} "
                       f"wrong={n['wrong']} recount={recount} buckets={buckets} idle={idle}")
    except Exception as e:
        r.fail("SPILL-45 race (setup)", f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    from helpers import SubTestResult
    r = SubTestResult()
    test_spill45_reaches_the_newest_put_and_round_trips_bit_exact(r)
    test_spill45_absent_and_already_spilled_are_both_false(r)
    test_spill45_no_disk_tier_leaves_the_frame_in_ram(r)
    test_spill45_races_get_and_put(r)
    r.summary()
