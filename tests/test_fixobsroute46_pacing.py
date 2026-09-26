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

PACE-462 (OVERHEAD-462's second cut) touches these same O4 rows: the cook's device is now
resolved to an INDEX and compared against `torch.cuda.current_device()` once, in `reset()`,
and `torch.cuda.device(...)` is entered only when that index is NOT already ambient-current
(skipping a ~5us context-manager round trip on the common single-GPU/already-on-device
case). `_DeviceSpy` below mocks `torch.cuda.current_device()` to a FIXED value (0) so O4's
own tests can keep proving "still entered for a genuinely non-current device" (`"cuda:1"`,
mocked-current 0) without that assertion becoming vacuous, and a new pair of tests proves
the skip fires when the cook's device IS mocked-current.
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
    no-op and `query()` always reports done, so `paced_check`'s wait loop never blocks.
    Accepts (and ignores) `blocking=` — PACE-462's ring creates its events with
    `torch.cuda.Event(blocking=True)`, so a mock with no constructor args at all raises
    `TypeError` the instant a real ring slot needs building, which is exactly what a
    CI/canonical run caught here before this fix."""

    def __init__(self, blocking=False):
        self.blocking = blocking

    def record(self):
        pass

    def query(self):
        return True


class _DeviceSpy:
    """Patches `torch.cuda.device` (the context manager), `torch.cuda.is_available`,
    `torch.cuda.current_device` and `torch.cuda.Event` well enough to drive `pacing.py`'s
    CUDA branch on ANY box, CUDA or not — this test's whole point is the MECHANISM (does the
    event get recorded inside `with torch.cuda.device(the cook's device):`, and is that
    entry skipped when the cook's device is already ambient-current?), never a real kernel or
    a real device mismatch, so it needs no `r.skip` and no real GPU (SIMP-3: "give the row a
    witness that runs without it" beats spending a skip-budget slot on a mechanism a mock can
    prove). `current` is the FIXED value `torch.cuda.current_device()` reports (default 0),
    fixed rather than tracked so a test can put a genuinely non-current index (e.g. `"cuda:1"`)
    on one side of the comparison and a genuinely current one (`"cuda:0"`/`"cuda"`) on the
    other, deliberately."""

    def __init__(self, current=0):
        self.calls = []
        self.current = current
        self._real_available = torch.cuda.is_available
        self._real_device_ctx = torch.cuda.device
        self._real_event = torch.cuda.Event
        self._real_current_device = torch.cuda.current_device

    def __enter__(self):
        spy = self

        def _fake_available():
            return True

        def _fake_current_device():
            return spy.current

        class _FakeDeviceCtx:
            def __init__(self, dev):
                spy.calls.append(dev)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        torch.cuda.is_available = _fake_available
        torch.cuda.current_device = _fake_current_device
        torch.cuda.device = _FakeDeviceCtx
        torch.cuda.Event = _FakeEvent
        return self

    def __exit__(self, *exc):
        torch.cuda.is_available = self._real_available
        torch.cuda.current_device = self._real_current_device
        torch.cuda.device = self._real_device_ctx
        torch.cuda.Event = self._real_event
        return False


def test_o4_paced_check_records_the_event_on_the_cooks_device(r):
    """O4: with a genuinely NON-current device (`"cuda:1"`, mocked-current 0), confirm
    `paced_check`'s event recording happens INSIDE a `with torch.cuda.device(<the cook's
    device>):` block, not on whatever device is ambient — the OVERHEAD-462 skip must not
    fire when the device really does differ from ambient-current."""
    print("\n--- FIX-OBSROUTE O4: paced_check records on the cook's device ---")
    with _DeviceSpy(current=0) as spy:
        _pace.reset(_NeverTrips(pace=True), "cuda:1")
        _pace.paced_check(_NeverTrips(pace=True), "cuda:1")  # first poll: no prev event to wait on
        _pace.paced_check(_NeverTrips(pace=True), "cuda:1")  # second poll: waits, then re-records
    if not spy.calls:
        r.fail("FIX-OBSROUTE O4 paced_check", "torch.cuda.device(...) was never entered — "
               "the event is still recorded on the ambient device")
    elif any(c != "cuda:1" for c in spy.calls):
        r.fail("FIX-OBSROUTE O4 paced_check", f"torch.cuda.device(...) was entered with "
               f"unexpected device(s): {spy.calls}")
    else:
        r.ok(f"paced_check entered torch.cuda.device('cuda:1') {len(spy.calls)} time(s), "
             f"naming the cook's own (non-current) device")


def test_o4_cook_done_event_records_the_event_on_the_requested_device(r):
    """Same mechanism check for `cook_done_event` (the `CookResult.done` fence), also on a
    genuinely non-current device."""
    print("\n--- FIX-OBSROUTE O4: cook_done_event records on the requested device ---")
    with _DeviceSpy(current=0) as spy:
        ev = _pace.cook_done_event("cuda:1")
    if ev is None:
        r.fail("FIX-OBSROUTE O4 cook_done_event", "returned None with CUDA mocked available")
    elif spy.calls != ["cuda:1"]:
        r.fail("FIX-OBSROUTE O4 cook_done_event", f"expected one torch.cuda.device('cuda:1') "
               f"entry, got {spy.calls}")
    else:
        r.ok("cook_done_event entered torch.cuda.device('cuda:1') exactly once")


def test_overhead462_paced_check_skips_device_ctx_when_already_current(r):
    """OVERHEAD-462: when the cook's device IS the mocked-current device, `paced_check`
    must NOT enter `torch.cuda.device(...)` at all — recording with no context switch is
    byte-identical, and the switch itself is the ~5us this cut removes."""
    print("\n--- OVERHEAD-462: paced_check skips torch.cuda.device(...) when current ---")
    with _DeviceSpy(current=0) as spy:
        _pace.reset(_NeverTrips(pace=True), "cuda:0")
        _pace.paced_check(_NeverTrips(pace=True), "cuda:0")
        _pace.paced_check(_NeverTrips(pace=True), "cuda:0")
    if spy.calls:
        r.fail("OVERHEAD-462 paced_check skip", f"torch.cuda.device(...) was entered "
               f"{len(spy.calls)} time(s) for an already-current device: {spy.calls}")
    else:
        r.ok("paced_check recorded on an already-current device with no context-manager entry")


def test_overhead462_cook_done_event_skips_device_ctx_when_already_current(r):
    """Same skip, for `cook_done_event`."""
    print("\n--- OVERHEAD-462: cook_done_event skips torch.cuda.device(...) when current ---")
    with _DeviceSpy(current=0) as spy:
        ev = _pace.cook_done_event("cuda:0")
    if ev is None:
        r.fail("OVERHEAD-462 cook_done_event skip", "returned None with CUDA mocked available")
    elif spy.calls:
        r.fail("OVERHEAD-462 cook_done_event skip", f"torch.cuda.device(...) was entered "
               f"{len(spy.calls)} time(s) for an already-current device: {spy.calls}")
    else:
        r.ok("cook_done_event recorded on an already-current device with no context-manager "
             "entry, and still returned an event")


def test_overhead462_cook_done_event_reuses_resets_cached_answer(r):
    """OVERHEAD-462 arm (a): after `reset()` has resolved is-CUDA/index/is-current for a
    device, `cook_done_event` for that SAME device must not re-derive it — `_is_cuda`
    (via `torch.device(...)` + `torch.cuda.is_available()`) is called at most once across
    `reset()` + N `cook_done_event` calls, not once per call."""
    print("\n--- OVERHEAD-462: cook_done_event reuses reset()'s cached answer ---")
    calls = {"n": 0}
    with _DeviceSpy(current=0) as spy:
        real_available = torch.cuda.is_available

        def _counting_available():
            calls["n"] += 1
            return real_available()

        torch.cuda.is_available = _counting_available
        try:
            _pace.reset(_NeverTrips(pace=False), "cuda:0")
            for _ in range(5):
                _pace.cook_done_event("cuda:0")
        finally:
            torch.cuda.is_available = real_available
    if calls["n"] == 1:
        r.ok("torch.cuda.is_available() was called exactly once across reset() + 5 "
             "cook_done_event calls for the same device")
    else:
        r.fail("OVERHEAD-462 cache reuse", f"torch.cuda.is_available() was called "
               f"{calls['n']} times (expected 1)")
    _ = spy


def test_overhead462_cook_done_event_falls_back_without_a_prior_reset(r):
    """The fallback half of arm (a): `cook_done_event` called with NO prior `reset()` on
    this thread (or a stale cache for a different device) must still compute the right
    answer fresh, exactly as it did before this cut."""
    print("\n--- OVERHEAD-462: cook_done_event falls back with no cached answer ---")
    with _DeviceSpy(current=0) as spy:
        _pace._state.__dict__.clear()  # noqa: SLF001 (simulate: never reset on this thread)
        ev = _pace.cook_done_event("cuda:1")
    if ev is None:
        r.fail("OVERHEAD-462 fallback", "returned None with CUDA mocked available")
    elif spy.calls != ["cuda:1"]:
        r.fail("OVERHEAD-462 fallback", f"expected one torch.cuda.device('cuda:1') entry via "
               f"the fresh-compute fallback, got {spy.calls}")
    else:
        r.ok("cook_done_event computed the right (non-current) answer with no reset() cache "
             "to read")


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
