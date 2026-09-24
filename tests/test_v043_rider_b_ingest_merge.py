"""v0.43.0 rider (b) — the ingest-event fence merge is a MECHANICAL MOVE ONLY.

`interpreter.py`'s inline ingest-event block and `compiled.py`'s (former, separately
defined) `_record_ingest_event` were two near-identical shapes of the same thing: record an
event on H2D ingest, then `.synchronize()` it, on the same stream. This merges exactly
those two shapes and nothing else — `streams.FrameHandle` and `ResultCache.pending_event`
stay untouched (see `DEVELOPMENT.md`'s rejected-decisions register for why).

This file is the behavior-identity half the existing ingest-event tests (`test_v020_phase1.py`
XPU-3/XPU-4, `test_v0422_race.py`, `test_codegen_param_device.py`, `test_perf7_compiled_cold.py`
— all re-run UNCHANGED by this move, per the ask) do not directly prove: that there is now
exactly ONE implementation, not two copies that happen to agree today.
"""
import torch

from helpers import *

from TEX_Wrangle.tex_runtime import interpreter, compiled

_CUDA = torch.cuda.is_available()


def test_rt_b_single_shared_ingest_event_helper(r: SubTestResult):
    print("\n--- RT-b: compiled.py and interpreter.py share ONE _record_ingest_event ---")
    try:
        if compiled._record_ingest_event is not interpreter._record_ingest_event:
            r.fail("RT-b merge",
                   "compiled._record_ingest_event is not the SAME function object as "
                   "interpreter._record_ingest_event — the merge left two implementations")
            return
        r.ok("compiled._record_ingest_event IS interpreter._record_ingest_event "
             "(one helper, not two copies)")
    except Exception as e:
        r.fail("RT-b merge", f"{type(e).__name__}: {e}")


def test_rt_b_ingest_event_behavior_identity(r: SubTestResult):
    print("\n--- RT-b: the merged helper's three answers, unit-level ---")
    if not _CUDA:
        r.skip("RT-b ingest behavior", "no CUDA on this box — the pinned H2D leg never fires")
        return
    try:
        dev = torch.device("cuda", torch.cuda.current_device())
        fn = interpreter._record_ingest_event

        # A pinned CPU tensor headed to CUDA records a real event.
        pinned = torch.rand(4, 4).pin_memory()
        ev = fn({"A": pinned}, dev)
        if ev is None:
            r.fail("RT-b behavior", "pinned CPU->CUDA binding did not record an event")
            return
        ev.synchronize()  # must not raise

        # A pageable (non-pinned) CPU tensor never records one.
        pageable = torch.rand(4, 4)
        if fn({"A": pageable}, dev) is not None:
            r.fail("RT-b behavior", "pageable CPU binding wrongly recorded an ingest event")
            return

        # A CPU device never records one, regardless of pinning.
        if fn({"A": pinned}, torch.device("cpu")) is not None:
            r.fail("RT-b behavior", "CPU device wrongly recorded an ingest event")
            return

        r.ok("pinned CPU->CUDA records+synchronizes; pageable and CPU-device do not")
    except Exception as e:
        r.fail("RT-b ingest behavior", f"{type(e).__name__}: {e}")
