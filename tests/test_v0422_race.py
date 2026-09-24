"""v0422-race (TRK-178) — a targeted repro attempt for the one CUDA-leg `wrong=2` red the
v0.42.0 release gate produced in `test_v033_cache8_touch_and_in_survive_a_threaded_race`, never
reproduced since (0/270 attempts across four contention shapes, per
`bug_reports/pending/v042-race.md`). That row races six kinds of traffic at once over 32x32
frames; this row isolates the ONE leg the audit named as leading suspect — `ResultCache._restore`'s
non-blocking pinned host-to-device copy on the disk-spill/restore leg — and gives it everything
that shape needs to show up: frames big enough that the DMA takes real time (`_pin_worthwhile`'s
band starts at 1 MiB; these are 64 MiB, so the copy is on the order of milliseconds, not
microseconds), and several reader threads released together the instant the frame becomes
servable again, so some of them race straight into `_restore` (a fresh disk-tier miss) while
others may catch the entry the instant it lands in `_ram`, mid-copy.

THE AUDIT (docs/worklog/v0422-race/handback.md) traced why this has been so hard to reproduce.
`_restore`'s H2D (`tex_results.py` around `pinned.to(dev, non_blocking=True)`) records no CUDA
event and does not synchronize before handing the tensor to `_admit` and back through `get`. But
nothing reachable from `get`/`put`/`_restore` ever pushes a non-default CUDA stream — grep the
package for `torch.cuda.stream(`/`set_stream`/`cuda.Stream(` outside `tex_runtime/graphed.py`'s
warm-up captures (which fence with `wait_stream` before and after and never touch `ResultCache`)
and outside `tex_runtime/streams.py`'s `egress` (the D2H demote leg, which already carries its
own event via `FrameHandle` — see there). Every thread in this test, like every thread in the
codebase's own `get`/`put` callers, is on the one CUDA default stream shared by the whole
process; CUDA orders work on one stream strictly by enqueue time regardless of which host thread
issued it, and the enqueue of `_restore`'s H2D happens-before `_admit` publishes the tensor
(under the cache's lock) happens-before any other thread's `get` can observe it — so a reader's
own kernel (`torch.equal`, enqueued on that same shared stream) cannot run ahead of the copy that
fills the buffer it reads. The pinned SOURCE side is covered independently: `torch`'s caching
host allocator defers reusing a pinned block until any stream event that touched it fires, which
is exactly the mechanism the `pinned.to(dev, non_blocking=True)` / `del pinned` pattern relies on
without keeping the source name alive — see the CPython-level `Tensor.to`/H2D-copy_ implementation
this file does not re-derive.

So the shape this row exists to check is: does that hold under real load, not just on paper? If
it does not, `torch.equal` catches it directly (this is not a proxy test — bit-exactness against
the frame that was actually written is the same check `wrong` counts in the six-way race)."""
import tempfile
import threading

import torch

from helpers import devices as _devices, make_gradient_frame as _frame
from TEX_Wrangle import tex_results

# `_PIN_MIN_BYTES` is 1 MiB (tex_marshalling.py) — comfortably below this, so the pinned/
# non_blocking leg is the one under test, not the plain `host.to(dev)` fallback. 2048x2048x4 fp32
# is 64 MiB: at typical laptop PCIe bandwidth the H2D alone is on the order of a few ms, wide
# enough for a handful of reader threads (started together, no stagger) to land inside it.
_RES = 2048
_RAM_BUDGET_MB = 512          # comfortably holds one 64 MiB frame with headroom to spare
_ITERATIONS = 20
_READERS = 8


def test_v0422_race_restore_pinned_h2d_survives_concurrent_readers(r):
    if "cuda" not in _devices():
        r.skip("v0422 restore race", "no CUDA on this box — the pinned H2D leg never fires")
        return
    with tempfile.TemporaryDirectory() as d:
        c = tex_results.ResultCache(cache_dir=d, budget_mb=_RAM_BUDGET_MB)
        errors = []
        restores_before = c.restores
        for it in range(_ITERATIONS):
            key = f"race{it}"
            frame = _frame(res=_RES, device="cuda", scale=1.0 + it * 0.001)
            # Demote to host RAM (room to spare — see test_v033_cache8's own note on why the
            # order matters: demoting BEFORE tightening the RAM budget is what keeps `home` !=
            # `device` distinct from the spill, so the frame is actually exercising the
            # demoted-THEN-spilled leg `_restore` reads back from, not a same-device write).
            c.set_vram_budget(0)
            c.put(key, frame)
            # `_queue_demotions` never demotes the MRU entry (the frame just cooked is the one
            # about to be read) — with only one entry in the cache that guard fires and `key`
            # never leaves CUDA. A throwaway second put makes `key` the LRU so it demotes.
            c.put(f"{key}-sentinel", _frame(res=32, device="cuda", scale=2.0))
            # Now force it out to disk: `_restore` is unreachable from a RAM hit.
            c.set_budget(0)
            # Reopen room and disarm residency before reading, so the restored frame lands on
            # its home CUDA device and stays there long enough for every reader to see it —
            # a still-armed ceiling of 0 would re-demote it the instant it landed.
            c.set_vram_budget(None)
            c.set_budget(_RAM_BUDGET_MB)

            barrier = threading.Barrier(_READERS)

            def reader(idx):
                barrier.wait()
                got = c.get(key, copy=False)
                if got is not None and not torch.equal(got, frame):
                    errors.append(f"iter={it} reader={idx}: get() served content "
                                  f"that does not match the frame written for {key!r}")

            threads = [threading.Thread(target=reader, args=(j,), daemon=True)
                       for j in range(_READERS)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30.0)
            hung = [t.name for t in threads if t.is_alive()]
            if hung:
                errors.append(f"iter={it}: reader thread(s) still alive: {hung}")
        restored = c.restores - restores_before
        ok = not errors and restored > 0
        r.ok(f"[cuda] {_ITERATIONS} spill/restore rounds x {_READERS} readers released "
             f"together ({restored} restores through the pinned H2D leg): 0 content "
             f"mismatches") if ok else \
            r.fail("v0422 restore race",
                   f"errors={errors[:5]} (of {len(errors)}) restored={restored}")
