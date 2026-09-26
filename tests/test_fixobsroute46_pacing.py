"""FIX-OBSROUTE (v0.46 Phase C) — O4/O5: `tex_runtime/pacing.py` findings.

  O4 [B2#3]: `paced_check`/`cook_done_event` recorded their CUDA event on whatever device
     was AMBIENT (`torch.cuda.current_device()`, generally device 0), not the cook's own
     `device` argument — wrong on any cook running on a non-default CUDA device. Fixed by
     wrapping the event creation in `with torch.cuda.device(device):`, the same discipline
     `tex_runtime/graphed.py:571` already uses to replay on the cook's device.
  O5 [R4#6, low]: `paced_check` re-derived "does this cook want pacing?"
     (`wants_pacing(token) and _is_cuda(device)`) on EVERY poll, for the cook's whole
     duration, instead of resolving it once. Fixed by resolving it in `reset()` — called
     once per cook — and reading the cached answer (`_state.paced`) from `paced_check`.
"""
import torch

from TEX_Wrangle.tex_runtime import pacing as _pace


class _NeverTrips:
    def __init__(self, pace=True):
        self.pace = pace

    def check(self):
        pass


class _FakeEvent:
    """A stand-in for `torch.cuda.Event` that needs no real CUDA context: `record()` is a
    no-op and `query()` always reports done, so `paced_check`'s wait loop never blocks."""

    def record(self):
        pass

    def query(self):
        return True


class _DeviceSpy:
    """Patches `torch.cuda.device` (the context manager) and `torch.cuda.is_available`/
    `torch.cuda.Event` well enough to drive `pacing.py`'s CUDA branch on ANY box, CUDA or
    not — this test's whole point is the MECHANISM (does the event get recorded inside
    `with torch.cuda.device(the cook's device):`?), never a real kernel or a real device
    mismatch, so it needs no `r.skip` and no real GPU (SIMP-3: "give the row a witness that
    runs without it" beats spending a skip-budget slot on a mechanism a mock can prove)."""

    def __init__(self):
        self.calls = []
        self._real_available = torch.cuda.is_available
        self._real_device_ctx = torch.cuda.device
        self._real_event = torch.cuda.Event

    def __enter__(self):
        spy = self

        def _fake_available():
            return True

        class _FakeDeviceCtx:
            def __init__(self, dev):
                spy.calls.append(dev)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        torch.cuda.is_available = _fake_available
        torch.cuda.device = _FakeDeviceCtx
        torch.cuda.Event = _FakeEvent
        return self

    def __exit__(self, *exc):
        torch.cuda.is_available = self._real_available
        torch.cuda.device = self._real_device_ctx
        torch.cuda.Event = self._real_event
        return False


def test_o4_paced_check_records_the_event_on_the_cooks_device(r):
    """O4: with `torch.cuda.device`/`is_available`/`Event` mocked (see `_DeviceSpy` — this
    needs no real CUDA device, only the CUDA-shaped code PATH to be reachable), confirm
    `paced_check`'s event recording happens INSIDE a `with torch.cuda.device(<the cook's
    device>):` block, not on whatever device is ambient."""
    print("\n--- FIX-OBSROUTE O4: paced_check records on the cook's device ---")
    with _DeviceSpy() as spy:
        _pace.reset(_NeverTrips(pace=True), "cuda")
        _pace.paced_check(_NeverTrips(pace=True), "cuda")   # first poll: no prev event to wait on
        _pace.paced_check(_NeverTrips(pace=True), "cuda")   # second poll: waits, then re-records
    if not spy.calls:
        r.fail("FIX-OBSROUTE O4 paced_check", "torch.cuda.device(...) was never entered — "
               "the event is still recorded on the ambient device")
    elif any(c != "cuda" for c in spy.calls):
        r.fail("FIX-OBSROUTE O4 paced_check", f"torch.cuda.device(...) was entered with "
               f"unexpected device(s): {spy.calls}")
    else:
        r.ok(f"paced_check entered torch.cuda.device('cuda') {len(spy.calls)} time(s), "
             f"naming the cook's own device")


def test_o4_cook_done_event_records_the_event_on_the_requested_device(r):
    """Same mechanism check for `cook_done_event` (the `CookResult.done` fence)."""
    print("\n--- FIX-OBSROUTE O4: cook_done_event records on the requested device ---")
    with _DeviceSpy() as spy:
        ev = _pace.cook_done_event("cuda")
    if ev is None:
        r.fail("FIX-OBSROUTE O4 cook_done_event", "returned None with CUDA mocked available")
    elif spy.calls != ["cuda"]:
        r.fail("FIX-OBSROUTE O4 cook_done_event", f"expected one torch.cuda.device('cuda') "
               f"entry, got {spy.calls}")
    else:
        r.ok("cook_done_event entered torch.cuda.device('cuda') exactly once")


def test_o5_paced_check_does_not_rederive_wants_pacing_per_poll(r):
    """O5: `wants_pacing` must be resolved once, by `reset()`, not once per `paced_check`
    poll. Runs on CPU — pre-fix, `paced_check` called `wants_pacing(token)` unconditionally
    on every call regardless of device (it's the LEFT operand of the `or` that gates the
    unpaced fast path), so this does not need a CUDA device to be red at base."""
    print("\n--- FIX-OBSROUTE O5: 'paced?' is resolved once, in reset() ---")
    calls = {"n": 0}
    real_wants_pacing = _pace.wants_pacing

    def _spy(token):
        calls["n"] += 1
        return real_wants_pacing(token)

    _pace.wants_pacing = _spy
    try:
        tok = _NeverTrips(pace=True)
        _pace.reset(tok, "cpu")             # exactly one call, if O5 landed
        for _ in range(10):
            _pace.paced_check(tok, "cpu")   # must read the cached answer, not re-derive it
    finally:
        _pace.wants_pacing = real_wants_pacing
    if calls["n"] == 1:
        r.ok("wants_pacing was called exactly once (by reset()), across reset() + 10 "
             "paced_check polls")
    else:
        r.fail("FIX-OBSROUTE O5", f"wants_pacing was called {calls['n']} times for one "
               f"reset() + 10 paced_check polls (expected exactly 1)")


def test_o5_reset_with_no_args_still_reads_as_unpaced(r):
    """Backward-compat: a caller that still calls `reset()` with no arguments (there should
    be none left in this tree, but the contract matters) gets the same answer the old
    per-poll recomputation gave a bare/absent token — unpaced."""
    print("\n--- FIX-OBSROUTE O5: reset() with no token defaults to unpaced ---")
    _pace.reset()
    tok = _NeverTrips(pace=True)
    calls = {"n": 0}

    def _counting_check():
        calls["n"] += 1

    tok.check = _counting_check
    _pace.paced_check(tok, "cuda" if torch.cuda.is_available() else "cpu")
    if calls["n"] == 1:
        r.ok("reset() with no arguments reads as unpaced (plain token.check(), no event wait)")
    else:
        r.fail("FIX-OBSROUTE O5 default", f"expected exactly one token.check() call, got "
               f"{calls['n']}")
