"""tex_results_residency.py — CACHE-8 residency ladder (split out of tex_results.py, SPLIT-R).

The VRAM -> host RAM half of the frame cache's tiering: above a configured VRAM ceiling, the
coldest CUDA-resident frames are DEMOTED to host RAM instead of spilled to disk (a demotion
costs 10.8 ms at 4K against a 204 ms spill, and the frame stays a cache hit either way); a hit
on a demoted frame PROMOTES it back home. Disarmed by default (`set_vram_budget(None)`), which
is v0.32's behaviour exactly — a frame cache that has never been told a VRAM budget must not
start moving frames between devices because it was upgraded.

`_ResultCacheResidency` is a MIXIN, not a standalone object: `tex_results.ResultCache` inherits
it, so every method here resolves as a `ResultCache` method exactly as it did before the split
(`self._promote(...)`, `cache.set_vram_budget(...)`, ...) and reads/writes the SAME instance
state (`_lock`, `_ram`, `_bytes_by_dev`, `_pending_demotes`, `_demoting`, `_vram_budget`,
`demotions`, `promotions`) that `tex_results.py` still declares in `ResultCache.__init__`. This
mixin owns no state of its own. `_dev_bucket` is the one plain function the ladder shares with
the rest of `ResultCache` (`_admit`, `_remove`, `governed_bytes`, `evict_bytes`, all still in
tex_results.py); `tex_results.py` re-imports it, so `tex_results._dev_bucket` keeps resolving
for every existing caller.
"""


def _dev_bucket(device) -> str:
    """The per-device accounting bucket for a device or device string. One spelling: the
    ternary was written out at each accounting site, and `_bytes_by_dev` only has meaning if
    every site agrees on which bucket an entry lands in. Non-CUDA accelerators (mps/xpu) bucket
    with cpu today — the governor arbitrates a CUDA pool and a host pool, and nothing else."""
    return "cuda" if str(device).startswith("cuda") else "cpu"


class _ResultCacheResidency:
    """CACHE-8: VRAM -> host RAM residency, mixed into `tex_results.ResultCache` (see that
    class's own docstring for the thread-safety and lock-scope rules these methods obey — they
    are unchanged by the move)."""

    # ── CACHE-8: residency (VRAM -> host RAM -> disk) ──
    def set_vram_budget(self, mb) -> None:
        """Arm the residency tier: above `mb` megabytes of CUDA-resident frames, the coldest
        are moved to host RAM. `None` disarms it, which is v0.32's behaviour exactly.

        Why a SECOND budget rather than a smarter single one: the two are different resources
        with different prices. `_budget` caps how much the cache holds AT ALL and its overflow
        goes to disk; this caps how much of that sits in VRAM, and its overflow goes to a place
        that is still a cache hit. A frame cache on a CUDA host is competing with the cook
        itself for VRAM while host RAM sits empty beside it — one number cannot express that.
        """
        with self._lock:
            self._vram_budget = None if mb is None else max(0, int(mb) * (1 << 20))
            if self._vram_budget is None:
                # A5 (v0.33.2): DISARM MEANS OFF, INCLUDING WORK ALREADY DECIDED ON. Victims
                # queued under the old ceiling would otherwise keep draining after this returns,
                # so a cache the host had just switched off went on moving frames cross-device —
                # contradicting the line above it ("`None` disarms it, which is v0.32's
                # behaviour exactly"; v0.32 never moves a frame between devices). Measured on
                # CUDA: `demotions` went 0 -> 1 with `vram_budget_bytes=None`.
                # Only the residency tier queues these (`evict_bytes` demotes solely when the
                # budget is set), so dropping the queue cannot strand a governor request.
                self._pending_demotes.clear()
            self._enforce_residency()
        self._drain_demotes()

    def _enforce_residency(self) -> None:
        """Queue the coldest CUDA frames for demotion until VRAM is under budget.

        POLICY, stated plainly because it is the part worth arguing with: victims are chosen
        by LRU, and a demoted frame is PROMOTED back on its next hit. That is a recency policy
        being used where the report asks for an access-FREQUENCY one, and the justification is
        that a frame cache's access pattern is a playhead — a scrub touches near-frames most
        recently and most often, so the two orderings largely coincide. `stats()` reports the
        demotion/promotion counts precisely so this can be revisited with a measurement instead
        of an opinion; a frequency-weighted victim choice is a change to this function alone.

        Never demotes the MRU entry (`len > 1` and the front-first walk), for the same reason
        `_enforce_ram_budget` does not: the frame just cooked is the one about to be read.

        Caller holds `_lock`."""
        if self._vram_budget is None:
            return
        over = self._bytes_by_dev["cuda"] - self._vram_budget
        if over > 0:                                  # O(1) early-out, the common case
            self._queue_demotions(over)

    def _queue_demotions(self, want: int) -> int:
        """Queue the coldest CUDA entries until ~`want` bytes are accounted for; returns the
        bytes queued. Caller holds `_lock`.

        THE one place a demotion victim is chosen. `_enforce_residency` (which wants VRAM back
        under a ceiling) and `evict_bytes` (which wants a byte count back for the governor)
        differ only in where that number comes from — and keeping the walk in one place is what
        makes `_enforce_residency`'s promise true when it says a frequency-weighted victim
        choice is a change to one function.

        Bytes are charged when a victim is QUEUED, not when the drain frees them: the drain is
        unconditional from here, and counting an entry twice would demote the world.

        A1: the skip-set is the queue UNION the in-flight set. A victim `_drain_demotes` has
        already popped is no longer in `_pending_demotes` but its copy has not landed — it is
        still on CUDA, still matches the walk, and re-queuing it gets it demoted TWICE. Both
        drains then commit the same cuda→cpu byte transfer, `_bytes_by_dev["cuda"]` goes
        NEGATIVE and stays skewed, `governed_bytes()` feeds the CACHE-5 governor garbage, and
        `_enforce_residency` stops firing forever because `over` can no longer be positive.
        Measured on a natural two-thread race over one 64 MiB frame: `{'cuda': -67107840}`
        against an actual 1024, with `demotions=2` for a single frame."""
        queued = {k for k, _e in self._pending_demotes} | self._demoting
        got = 0
        # A7: the MRU entry is the frame just cooked — the one about to be read. Excluding it
        # by KEY is what the docstring always promised; the `len(self._ram) <= 1` guard it used
        # to rely on is dead code here, because this function never removes an entry, so on any
        # multi-entry cache the walk reached the newest frame whenever `want` covered the older
        # CUDA bytes. Deterministic at `set_vram_budget(0)`: two puts demoted BOTH, including
        # the one whose `put` had just returned, which is then promoted back on its next hit —
        # ~22 ms of pointless copies per cook at 4K.
        mru = next(reversed(self._ram), None) if self._ram else None
        for key in list(self._ram.keys()):            # oldest -> newest
            if got >= want:
                break
            if key == mru:
                continue
            entry = self._ram.get(key)
            if entry is None or _dev_bucket(entry.device) != "cuda" or key in queued:
                continue
            self._pending_demotes.append((key, entry))
            queued.add(key)
            got += entry.nbytes
        return got

    def _drain_demotes(self) -> None:
        """Perform the queued D2H copies and swap the host buffers in. Called by the PUBLIC
        methods, after they release the lock.

        Unlike a spill, a demotion must leave the frame SERVABLE throughout — so the entry is
        never removed. It stays in `_ram`, on CUDA, answering `get` with the right pixels until
        the host copy exists; only then is slot 0 swapped and the per-device accounting moved.
        A concurrent `get` during the copy therefore serves the VRAM master, which is correct
        and is why this is not a window anyone has to reason about.

        XPU-2 is deliberately used in its `retained=True` mode here. A demoted frame's host
        buffer is RETAINED — for as long as the entry lives — and a page-locked buffer of that
        lifetime is a slow leak of unswappable memory. Copying into pinned and then cloning to
        pageable to release the lock would cost a second full host memcpy of the frame,
        which is more than the asynchrony saves on a copy that has almost nothing to overlap
        with. The handle is still the seam: when v0.34's async-write path hands a demoted frame
        to a writer thread, it does so through this same object.

        A demotion that fails leaves the frame exactly where it was — over budget, and correct.
        The budget is a target; the pixels are not."""
        import torch
        from .tex_runtime.streams import egress
        if self._lock.depth:
            return                # a composite (patch_region) holds the lock; it drains after
        while True:
            with self._lock:
                if not self._pending_demotes:
                    return
                key, entry = self._pending_demotes.popleft()
                if self._ram.get(key) is not entry or _dev_bucket(entry.device) != "cuda":
                    continue                          # evicted, replaced, or already demoted
                # A1: OFF the queue but not yet committed — `_queue_demotions` must still see it
                # as spoken for, or it re-queues a frame that is still on CUDA and a second drain
                # commits the same byte transfer.
                self._demoting.add(key)
                src = entry.tensor
            try:
                # Born frozen: slot 0's contract is a FROZEN master (`_spill` reads it as one).
                with torch.inference_mode():
                    host = egress(src, retained=True).tensor()
            except Exception:
                with self._lock:
                    self._demoting.discard(key)
                continue                              # leave it in VRAM; correctness is unharmed
            with self._lock:
                self._demoting.discard(key)
                cur = self._ram.get(key)
                # A1: identity AND device, mirroring `_promote`. Identity alone is not enough —
                # two drains can hold the SAME entry object, both find it unchanged, and both
                # apply the transfer. The device is what says whether the move already happened,
                # and it is the field the transfer itself mutates, so re-reading it under the
                # lock is the check that cannot be raced.
                if cur is not entry or _dev_bucket(cur.device) != "cuda":
                    continue                          # it moved on while we copied: drop the copy
                if self._vram_budget is None:
                    # A5, and LOAD-BEARING rather than belt-and-braces: `set_vram_budget(None)`
                    # empties `_pending_demotes`, but a victim THIS drain already popped is in
                    # neither the queue nor the clear's reach — it is mid-`egress`, holding only
                    # a local reference. Same argument `clear()` spells for spills ("a victim
                    # `_drain_spills` ALREADY popped is not in either queue"). Delete this and
                    # the disarm still lets one frame cross devices with the tier off.
                    continue
                self._bytes_by_dev["cuda"] -= entry.nbytes
                self._bytes_by_dev["cpu"] += entry.nbytes
                cur.tensor = host
                cur.device = "cpu"
                cur.pending_event = None   # TRK-178: described the replaced tensor, not `host`
                self.demotions += 1

    def _promote(self, key: str, entry):
        """Move a demoted frame back to its home device and return the promoted master, or the
        entry's current tensor if it cannot be moved.

        Called from `get` on a hit, and from `touch_promote` on a hint (CACHE-11) that is never
        counted as one — both OUTSIDE the lock, since this is an H2D copy (11.1 ms at 4K),
        exactly the class of work the lock rule excludes. The re-entry check under the lock is
        what makes that safe: if the entry changed while we copied, the copy is discarded."""
        # A5(d) WITHDRAWN in v0.33.2 — the lock-depth early-out that stood here is deferred, not
        # forgotten (DEVELOPMENT.md carries the row). It read:
        #
        #     if self._lock.depth: return entry.tensor
        #
        # and its argument was sound as far as it went: the drains refuse to run a full-frame
        # copy while a composite holds the lock, and `_promote` is an H2D (11.1 ms at 4K) that
        # `patch_region` reaches at depth 1. What it missed is that `_promote` is not a drain.
        # A drain DEFERS — the queue survives and `patch_region` runs it on release. This
        # DEGRADED: it silently handed back the host copy, and there is no `_pending_promotes`
        # to make good on it. Measured consequence, not a worry:
        #
        #     base demoted to host RAM (home=cuda:0) -> patch_region -> result device=cpu,
        #     result HOME=cpu, promotions=0
        #
        # `frame[...] = patch` accepts a CUDA source into a CPU destination (`copy_` is
        # cross-device), so nothing raises; `_admit` then records `home="cpu"` for the fresh
        # destination key because that is where the frame it was handed actually lives. The
        # patched frame has left the residency ladder permanently, and every stage downstream of
        # it inherits a CPU home — precisely the "one-way trip to the CPU" `_Entry.home` was
        # introduced to prevent. Trading a latency problem for a residency-correctness one is a
        # bad trade, which is why the early-out came out and stays out.
        #
        # The OBVIOUS repair is to make the early-out defer rather than degrade — queue the
        # entry on `_pending_promotes` and let a public method run the copy after releasing the
        # lock, exactly as `_drain_demotes` does for the other direction. It does not work.
        # CF-1 (v0.35) built the queue and then did NOT arm the early-out, because
        # the A5 pin showed the deferral targets the wrong frame: queueing the BASE is useless
        # — by the time the drain runs the base has been read and patched — and deferring moves
        # the patch arithmetic onto the host, so the RESULT lands host-resident. The pin asserts
        # `device == cuda` as well as `home == cuda`, and it is right to: a patched frame that
        # is merely homed to CUDA while sitting on the host has not come back.
        #
        # The fix that works is to resolve the base BEFORE `patch_region` takes the composite
        # lock, so `_promote` runs at depth 0 like every other promotion and the patch happens
        # on CUDA. That is a lock-SCOPE change entangled with `tag_key`'s `base is None` branch
        # (the A4 quality ratchet reads it) and it needs its own argument, not a ride on a
        # carry-forward item. The queue and its drain were BUILT for this and then removed
        # again in the same session rather than shipped: nothing appended to them, so they were
        # unreachable code claiming a guarantee, and the victim-walk guards that read them could
        # not be killed by a mutation row. What CF-1 keeps is the residency half — home
        # propagation through the nested put — which is what closes the one-way trip to the CPU.
        #
        # So the H2D below runs at whatever depth the caller reached it at: 0 from `get`, 1 when
        # `patch_region` holds the composite lock. The commit block re-acquires the RLock, which
        # is re-entrant and therefore correct at both; what depth 1 costs is latency only.
        import torch
        src, home = entry.tensor, entry.home
        try:
            with torch.inference_mode():
                dev = torch.device(home)
                moved = torch.empty(src.shape, dtype=src.dtype, device=dev)
                moved.copy_(src)
        except Exception:
            return src                                # stay on the CPU; a hit is still a hit
        with self._lock:
            cur = self._ram.get(key)
            if cur is not entry:
                # The entry was replaced or evicted while we copied. `moved` still holds THIS
                # entry's pixels at THIS entry's representation, which is what `get`'s captured
                # `orig_dtype` describes — handing back the new entry's tensor instead would
                # pair one frame's bytes with another frame's unpack.
                return moved
            if cur.device == home:
                return cur.tensor                     # another thread promoted it first
            self._bytes_by_dev[_dev_bucket(cur.device)] -= cur.nbytes
            self._bytes_by_dev[_dev_bucket(home)] += cur.nbytes
            cur.tensor = moved
            cur.device = home
            self.promotions += 1
            return moved
