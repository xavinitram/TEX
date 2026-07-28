"""v0.33.1 — the v0.33.0 release-audit findings, each pinned by a row that fails pre-fix.

All eight live behind ARMED paths (residency via `set_vram_budget`/GOV-1, the spill tier, or
opt-in `storage="uint16"`); none is reachable from the default ComfyUI cook. That is why this
is a patch release and not a recall — but "unreachable today" is a property of the wiring, not
of the code, so each one is closed and pinned rather than documented as unlikely.

  A1  a doubly-demoted frame drove `_bytes_by_dev["cuda"]` NEGATIVE and permanently disabled
      residency (the governor was then reading garbage)
  A2  `get()` served fp16 bytes to a caller owed fp32, through a second lock acquisition
  A3  the P0-6 residual — a fresh cache (`_spilled is None`, the reattach case) still lost a
      frame that spilled during `reindex_disk`'s scan
  A4  uint16 frames were destroyed by the spill tier: written, indexed, and unloadable
  A5  `clear(disk=True)` lost to an in-flight spill, resurrecting a cleared frame
  A6  `_restore`'s stale-epoch cleanup mutated the index unlocked
  A7  the "never demote the MRU" guard was dead code — the just-cooked frame was demoted
  A8  fp64-cooked packed frames restored as raw fp16
"""
import os
import tempfile
import threading

import torch

from helpers import devices as _devices, make_gradient_frame as _frame
from TEX_Wrangle import tex_packing, tex_results


def _recount(c):
    """The per-device totals recomputed from the entries themselves — the ground truth
    `_bytes_by_dev` is supposed to be a maintained view of."""
    out = {"cuda": 0, "cpu": 0}
    for e in c._ram.values():
        out[tex_results._dev_bucket(e.device)] += e.nbytes
    return out


# ── A1 ────────────────────────────────────────────────────────────────────────

def test_v0331_a1_double_demotion_cannot_skew_the_byte_totals(r):
    """Two threads racing `evict_bytes` over ONE frame demoted it twice, and both commits
    applied the cuda→cpu transfer: `_bytes_by_dev["cuda"]` went to -67107840 against an actual
    1024. That is not a cosmetic drift — `governed_bytes()` is what the CACHE-5 governor reads,
    and a negative `over` means `_enforce_residency` can never fire again.

    Two holes, both closed: the commit block re-checked entry IDENTITY but not DEVICE (unlike
    `_promote`, which checks both), and a victim popped off `_pending_demotes` was invisible to
    `_queue_demotions`, so it could be re-queued while its copy was still in flight."""
    if "cuda" not in _devices():
        r.ok("A1: double-demotion SKIPPED (no CUDA — nothing to demote)")
        return
    from TEX_Wrangle.tex_runtime import streams
    with tempfile.TemporaryDirectory() as d:
        c = tex_results.ResultCache(cache_dir=d)
        c.put("a", _frame(res=128, device="cuda"))
        c.put("b", _frame(res=128, device="cuda", scale=0.5))
        with c._lock:
            c._vram_budget = 0                        # arm without enforcing/draining

        # DETERMINISTIC, not a hopeful race. Block the first drain INSIDE its copy — exactly
        # the window where the victim is off `_pending_demotes` and still on CUDA — and run a
        # second drain from this thread. Pre-fix, the second drain's walk re-queues the same
        # frame and both commits apply the cuda→cpu transfer.
        inside, release = threading.Event(), threading.Event()
        real_egress = streams.egress

        def blocking_egress(src, **kw):
            inside.set()
            release.wait(5)
            return real_egress(src, **kw)

        streams.egress = blocking_egress
        t = threading.Thread(target=c._drain_demotes, daemon=True)
        try:
            with c._lock:
                c._queue_demotions(1 << 30)
            t.start()
            inside.wait(5)
            streams.egress = real_egress              # the second drain runs unblocked
            with c._lock:
                c._queue_demotions(1 << 30)           # the re-queue the fix must prevent
            c._drain_demotes()
            release.set()
            t.join(timeout=5)
        finally:
            streams.egress = real_egress
        with c._lock:
            got, truth = dict(c._bytes_by_dev), _recount(c)
            demotions = c.demotions
        negative = any(v < 0 for v in got.values())
        ok = got == truth and not negative
        r.ok(f"A1: {demotions} demotion(s), totals match a per-entry recount {got}") if ok \
            else r.fail("A1 byte skew",
                        f"maintained={got} recount={truth} negative={negative} "
                        f"demotions={demotions}")


def test_v0331_a1_a_demoting_frame_is_not_requeued(r):
    """The in-flight half, asserted directly: a key popped for demotion but not yet committed
    must be invisible to the victim walk. Without it the same frame is queued twice and two
    drains commit one transfer."""
    if "cuda" not in _devices():
        r.ok("A1: in-flight visibility SKIPPED (no CUDA)")
        return
    with tempfile.TemporaryDirectory() as d:
        c = tex_results.ResultCache(cache_dir=d)
        c.put("a", _frame(res=64, device="cuda"))
        c.put("b", _frame(res=64, device="cuda", scale=0.5))
        # Arm WITHOUT `set_vram_budget`, which enforces and drains immediately — the frames must
        # still be on CUDA for the victim walk to have anything to pick.
        with c._lock:
            c._vram_budget = 0
            c._queue_demotions(1 << 30)
            key, entry = c._pending_demotes.popleft()
            c._demoting.add(key)                      # simulate the drain's in-flight window
            before = len(c._pending_demotes)
            c._queue_demotions(1 << 30)               # must NOT re-queue `key`
            requeued = any(k == key for k, _e in c._pending_demotes)
            after = len(c._pending_demotes)
            c._demoting.discard(key)
        r.ok("A1: a frame whose copy is in flight is never re-queued") if not requeued else \
            r.fail("A1 requeue", f"{key} re-queued ({before} -> {after})")


# ── A2 ────────────────────────────────────────────────────────────────────────

def test_v0331_a2_restore_returns_the_representation_atomically(r):
    """`get` used to re-look-up `orig_dtype` in a SECOND lock acquisition after `_restore`
    returned. A concurrent `clear()`/eviction in that window drops the entry, the lookup reads
    `None`, and an fp16-STORED frame is served at storage dtype — breaking the one guarantee the
    storage tier makes ("a frame goes in fp32 and comes out fp32"). Reproduced without any
    patching, on the first iteration.

    The structural fix: `_restore` returns `(frame, orig_dtype)` from its own locked re-admit.
    This row pins the shape; the race row below pins the behaviour."""
    with tempfile.TemporaryDirectory() as d:
        c = tex_results.ResultCache(cache_dir=d, budget_mb=0)
        f = _frame(res=64)
        c.put("p", f, quality=tex_packing.PREVIEW)
        c.put("evictor", f)                            # forces "p" to disk
        got = c._restore("p")
        pair = isinstance(got, tuple) and len(got) == 2
        ok = pair and got[0] is not None and got[1] is torch.float32
        r.ok("A2: _restore hands back (frame, orig_dtype) as one answer") if ok else \
            r.fail("A2 restore shape", f"returned {type(got).__name__}: {got!r}")
        c.clear(disk=True)


def test_v0331_a2_a_racing_clear_never_serves_storage_dtype(r):
    """The behaviour, raced deliberately with a widened switch interval. `get` must return fp32
    or nothing — never the fp16 bytes."""
    with tempfile.TemporaryDirectory() as d:
        c = tex_results.ResultCache(cache_dir=d, budget_mb=0)
        f = _frame(res=64)
        c.put("p", f, quality=tex_packing.PREVIEW)     # stored fp16
        c.put("evictor", f)                            # forces "p" to disk

        # DETERMINISTIC. `clear()` is fired at precisely the moment the pre-fix code would have
        # re-looked-up `orig_dtype` — between `_restore`'s re-admit and `get`'s second lock
        # acquisition — by blocking inside the last thing `_restore` does. A sleep-and-hope race
        # proves nothing on a fast box; this fires every run.
        real_admit = tex_results.ResultCache._admit

        def clearing_admit(self, *a, **kw):
            entry = real_admit(self, *a, **kw)
            if kw.get("home") is not None or len(a) >= 4:
                self.clear()                           # the entry vanishes right here
            return entry

        tex_results.ResultCache._admit = clearing_admit
        try:
            out = c.get("p")
        finally:
            tex_results.ResultCache._admit = real_admit
        leaked = out is not None and out.dtype is not torch.float32
        r.ok(f"A2: a clear() landing mid-restore yields "
             f"{'a miss' if out is None else 'float32'}, never storage dtype") if not leaked \
            else r.fail("A2 dtype leak", f"served {out.dtype} to a caller owed float32")
        c.clear(disk=True)


# ── A3 ────────────────────────────────────────────────────────────────────────

def test_v0331_a3_reindex_never_rebinds_over_a_racing_spill(r):
    """THE P0-6 RESIDUAL, and the v0.33 pin for it was decorative — it passed against the
    pre-fix rebind. A fresh cache (`_spilled is None`) is exactly `reindex_disk`'s own reason to
    exist (ENG-13 reattach), and there `_spill` skips recording, the merge finds "nothing
    missed", and the tail rebinds to a definite set EXCLUDING the racing frame. That frame is
    unserveable forever: `_restore` short-circuits on the membership set without stat-ing.

    Driven for real: `os.scandir` is wrapped so a genuine spill lands between the scan and the
    merge, and membership starts UNKNOWN — the case the previous pin never exercised."""
    with tempfile.TemporaryDirectory() as d:
        c = tex_results.ResultCache(cache_dir=d, budget_mb=0)
        c.put("seed", _frame(res=32))
        c.put("evict-seed", _frame(res=32))            # something already on disk
        with c._lock:
            c._spilled = None                          # the fresh/reattach state
        real_scandir = os.scandir
        calls = {"n": 0}

        def racing_scandir(path):
            # FIRE ON THE MEMBERSHIP WALK, NOT THE SWEEP. `reindex_disk` calls `sweep_temps()`
            # first, which scandirs the same directory — a first draft of this row injected
            # there, so the spill landed BEFORE the walk, the walk saw it, and the row passed
            # against the pre-fix rebind. That is the same decorative-pin failure this test
            # exists to correct, reproduced while writing it.
            calls["n"] += 1
            it = real_scandir(path)
            if calls["n"] == 2:
                c.put("racer", _frame(res=32))
                c.put("evict-racer", _frame(res=32))
            return it

        os.scandir = racing_scandir
        try:
            c.reindex_disk()
        finally:
            os.scandir = real_scandir
        served = c.get("racer")
        ok = served is not None
        r.ok("A3: a frame spilled during the reindex scan is still servable "
             f"(membership left {'unknown' if c._spilled is None else 'definite'})") if ok \
            else r.fail("A3 reattach race",
                        f"the racing frame is unserveable; _spilled={c._spilled}")
        c.clear(disk=True)


# ── A4 ────────────────────────────────────────────────────────────────────────

def test_v0331_a4_every_storage_dtype_survives_the_spill_tier(r):
    """torch 2.10 pickles `torch.uint16` and cannot load it back, so the spill "succeeded" while
    the frame was permanently lost and its `.frame` leaked the disk budget forever (even the
    stale-epoch cleanup is unreachable — the load raises first). fp16 was already pinned;
    uint16 was the hole, so this row covers the whole storage vocabulary rather than one dtype."""
    bad = []
    for label, kw in (("as-cooked", {}),
                      ("fp16", {"quality": tex_packing.PREVIEW}),
                      ("uint16", {"quality": tex_packing.PREVIEW, "storage": "uint16"})):
        with tempfile.TemporaryDirectory() as d:
            c = tex_results.ResultCache(cache_dir=d, budget_mb=0)
            f = _frame(res=64)
            c.put("k", f, **kw)
            c.put("evictor", f)                        # forces "k" out to disk
            got = c.get("k")
            if got is None:
                bad.append(f"{label}: LOST on eviction (restores={c.restores})")
            elif got.dtype is not torch.float32:
                bad.append(f"{label}: served {got.dtype}")
            elif float((got - f).abs().max()) > 4e-3:
                bad.append(f"{label}: maxdiff {float((got - f).abs().max()):.2e}")
            c.clear(disk=True)
    r.ok("A4: every storage representation round-trips the disk tier") if not bad else \
        r.fail("A4 spill round-trip", "; ".join(bad))


def test_v0331_a4_frame_records_carry_a_format_version(r):
    """B5a. v0.33 changed the `.frame` record (adding `orig`, redefining `device` as HOME)
    without a version field; compat held only because `rec.get("orig")` defaults safely and a
    v0.32 frame's device and home were necessarily equal. Neither accident survives the next
    change. Absence still reads as v0-raw, so both formats are readable for one release."""
    import pickle
    with tempfile.TemporaryDirectory() as d:
        c = tex_results.ResultCache(cache_dir=d, budget_mb=0)
        f = _frame(res=32)
        c.put("k", f)
        c.put("evictor", f)
        with open(c._disk_path("k"), "rb") as fh:
            rec = pickle.load(fh)
        # ...and a v0 record (no `fmt`, no `orig`) must still restore.
        v0 = {"t": f[0].clone(), "device": "cpu", "canvas": None,
              "epoch": tex_results.env_epoch()}
        with open(c._disk_path("legacy"), "wb") as fh:
            pickle.dump(v0, fh, protocol=pickle.HIGHEST_PROTOCOL)
        with c._lock:
            c._spilled = None
        legacy = c.get("legacy")
        ok = rec.get("fmt") == tex_results._FRAME_FORMAT and legacy is not None
        r.ok(f"A4/B5a: records carry fmt={rec.get('fmt')}; a v0 record still restores") if ok \
            else r.fail("A4 format version",
                        f"fmt={rec.get('fmt')} legacy_restored={legacy is not None}")
        c.clear(disk=True)


# ── A5 / A7 / A8 ──────────────────────────────────────────────────────────────

def test_v0331_a5_clear_is_not_undone_by_an_inflight_spill(r):
    """A victim `_drain_spills` already popped is in neither queue, so `clear()` cannot cancel
    it: its write lands after the unlink loop, re-creating the file and re-indexing the key, and
    a later `get` serves the frame the user cleared. A generation counter, checked under the
    lock before the index add, makes the stale writer drop its own file instead."""
    with tempfile.TemporaryDirectory() as d:
        c = tex_results.ResultCache(cache_dir=d, budget_mb=0)
        f = _frame(res=32)
        # The writer must have ENTERED `_spill` (and captured its generation) BEFORE `clear()`
        # runs — that is the whole race. Block it inside the pickle write and clear from the
        # main thread; calling `_spill` after the clear reproduces nothing, because it would
        # capture the post-clear generation and be entitled to index.
        entry = tex_results._Entry(f, 0, f.numel() * 4, "cpu", None, None, "cpu")
        inside, release = threading.Event(), threading.Event()
        real_pickle = tex_results._atomic_pickle

        def blocking_pickle(path, data):
            inside.set()
            release.wait(5)
            return real_pickle(path, data)

        tex_results._atomic_pickle = blocking_pickle
        t = threading.Thread(target=c._spill, args=("k", entry), daemon=True)
        try:
            t.start()
            inside.wait(5)                             # the writer holds the OLD generation
            c.clear(disk=True)                         # the user clears everything
            release.set()
            t.join(timeout=5)
        finally:
            tex_results._atomic_pickle = real_pickle
        served = c.get("k")
        indexed = c._spilled is not None and "k" in c._spilled
        on_disk = os.path.exists(c._disk_path("k"))
        ok = served is None and not indexed and not on_disk
        r.ok("A5: a spill in flight across clear() does not resurrect the frame") if ok else \
            r.fail("A5 clear race",
                   f"served={served is not None} indexed={indexed} file_left={on_disk}")
        c.clear(disk=True)


def test_v0331_a7_the_mru_frame_is_never_demoted(r):
    """The guard the docstring promised was dead code: `_queue_demotions` never removes an
    entry, so `len(self._ram) <= 1` cannot fire on a multi-entry cache, and the oldest-first
    walk reached the just-cooked frame whenever `want` covered the older CUDA bytes. At
    `set_vram_budget(0)` that is fully deterministic — the frame whose `put` had just returned
    was demoted, then promoted back on its next hit: ~22 ms of pointless copies per cook at 4K."""
    if "cuda" not in _devices():
        r.ok("A7: MRU guard SKIPPED (no CUDA)")
        return
    with tempfile.TemporaryDirectory() as d:
        c = tex_results.ResultCache(cache_dir=d)
        c.set_vram_budget(0)
        c.put("a", _frame(res=64, device="cuda"))
        c.put("b", _frame(res=64, device="cuda", scale=0.5))
        mru = c._ram["b"]
        st = c.stats()
        ok = mru.device == mru.home and st["demotions"] >= 1
        r.ok(f"A7: the MRU frame stays resident while {st['demotions']} older one(s) demote") \
            if ok else r.fail("A7 MRU demoted",
                              f"mru device={mru.device} home={mru.home} "
                              f"demotions={st['demotions']}")


def test_v0331_a8_only_fp32_sources_are_packed(r):
    """`choose_storage` admitted any float wider than two bytes, so fp64 packed with
    `orig_dtype=torch.float64` — which the spill record's dtype table cannot spell, so `orig`
    wrote `None` and the restored entry FORGOT it was packed: float64 before eviction, float16
    after. Narrowing the source to fp32 fails toward exactness, and fp32 is the only dtype a
    TEX cook produces."""
    P = tex_packing.PREVIEW
    f32, f64 = _frame(res=32), _frame(res=32).double()
    rows = {"fp32": tex_packing.choose_storage(f32, quality=P),
            "fp64": tex_packing.choose_storage(f64, quality=P),
            "fp16-source": tex_packing.choose_storage(f32.half(), quality=P)}
    ok = (rows["fp32"] == tex_packing.FP16 and rows["fp64"] is None
          and rows["fp16-source"] is None)
    r.ok("A8: only an fp32 source packs; fp64 and an already-half frame decline") if ok else \
        r.fail("A8 source dtype", f"{rows}")


def test_v0331_a1_a_duplicate_queue_entry_commits_once(r):
    """The DEVICE re-check, exercised directly. `_demoting` closes the known route to a
    duplicate, so a black-box test cannot produce one any more — which made the mutation
    "stop re-checking the device" SURVIVE. The guard is still load-bearing: it is what makes a
    duplicate arriving by ANY future route harmless, and it is the same check `_promote` has
    always had.

    So the duplicate is injected. Identity alone cannot tell these apart — both queue entries
    are the same object — and only re-reading the device (which the transfer itself mutates)
    says the move already happened."""
    if "cuda" not in _devices():
        r.ok("A1: duplicate-commit SKIPPED (no CUDA)")
        return
    with tempfile.TemporaryDirectory() as d:
        c = tex_results.ResultCache(cache_dir=d)
        c.put("a", _frame(res=64, device="cuda"))
        c.put("b", _frame(res=64, device="cuda", scale=0.5))
        with c._lock:
            c._vram_budget = 0
            entry = c._ram["a"]
            c._pending_demotes.append(("a", entry))    # the same victim, queued twice
            c._pending_demotes.append(("a", entry))
        c._drain_demotes()
        with c._lock:
            got, truth = dict(c._bytes_by_dev), _recount(c)
            n = c.demotions
        ok = got == truth and n == 1 and all(v >= 0 for v in got.values())
        r.ok(f"A1: a doubly-queued victim commits exactly once ({n}), totals {got}") if ok \
            else r.fail("A1 duplicate commit",
                        f"demotions={n} maintained={got} recount={truth}")


def test_v0331_a3_a_learned_membership_set_also_survives_the_scan(r):
    """The `raced` guard specifically. The `unknown_at_entry` arm covers a FRESH cache; this
    one covers a cache whose membership is already a learned set, where the only witness that
    the scan is stale is the spill counter having moved. Without it the merge is computed
    against a `found` that predates the racing spill and the tail rebinds a definite set."""
    with tempfile.TemporaryDirectory() as d:
        c = tex_results.ResultCache(cache_dir=d, budget_mb=0)
        c.put("seed", _frame(res=32))
        c.put("evict-seed", _frame(res=32))
        c._learn_spilled()                             # membership is now a LEARNED set
        assert c._spilled is not None
        real_scandir = os.scandir
        calls = {"n": 0}

        def racing_scandir(path):
            calls["n"] += 1
            it = real_scandir(path)
            if calls["n"] == 2:                        # the membership walk, not sweep_temps
                c.put("racer", _frame(res=32))
                c.put("evict-racer", _frame(res=32))
            return it

        os.scandir = racing_scandir
        try:
            c.reindex_disk()
        finally:
            os.scandir = real_scandir
        served = c.get("racer")
        r.ok("A3: a learned membership set is not rebound over a racing spill either") \
            if served is not None else \
            r.fail("A3 learned-set race",
                   f"the racing frame is unserveable; _spilled={c._spilled}")
        c.clear(disk=True)
