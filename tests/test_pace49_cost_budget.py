"""PACE-49 -- a MEASURED per-call-site device-time budget, additive to `depth`. PACE-47e's
own `_HEAVY_PIXEL_THRESHOLD`/registry `heavy` tag are both proxies (a NAME or a CONSTANT); two
real shapes escape both (a `for`-loop whose body is expensive but below the pixel threshold; a
noise call whose runtime `octaves` makes it far more expensive than its binary `heavy` tag
says). This file is the mechanism half: deterministic, no real CUDA device and no wall-clock,
using `tex_testkit`'s `DeviceSpy`/`FakeCudaEvent` (the same scaffold
`test_pace462_bounded_lookahead.py` uses) plus a controlled fake host clock, so the cost
table's exact bookkeeping -- when it feeds, from which event pair, how many samples before it
is trusted, when it forces a fall-through -- is provable without hardware.
`benchmarks/preempt_drain_bench.py`'s own `loopbody256`/`fbmoctaves256` sweep rows are the
real-device acceptance shape.

Every row here is RED against base `7477a93` (v0.48.0): that `pacing.py` has no
`_COST_TABLE`, no `pace_budget_ms`, and `paced_check` takes no `call_site_id` keyword at all
(TypeError) -- so every assertion below has nothing matching to read.
"""
import pytest

from TEX_Wrangle.tex_runtime import pacing as _pace
from TEX_Wrangle.tex_testkit import DeviceSpy, FakeCudaEvent


@pytest.fixture(autouse=True)
def _fresh_pacing_state():
    """Same isolation `test_pace462_bounded_lookahead.py` uses: `pacing._state` is
    thread-local and persists across cooks on this SAME thread by design, so without this
    fixture one test's pool/cost-table state would leak into the next."""
    _pace._state.__dict__.clear()
    _pace._COST_TABLE.clear()
    yield
    _pace._state.__dict__.clear()
    _pace._COST_TABLE.clear()


class _Token:
    """Stride disabled by default is deliberately NOT the default here (unlike
    `test_pace462_bounded_lookahead.py`'s own `_Token`): PACE-49's own mechanism lives
    entirely inside the stride-economization branch, so these rows need striding armed. A
    generous default stride (10ms) plus a controlled fake clock keeps every poll inside the
    same window unless a test explicitly advances past it."""

    def __init__(self, pace=True, pace_depth=8, pace_stride_ms=10.0, pace_budget_ms=None):
        self.pace = pace
        self.pace_depth = pace_depth
        self.pace_stride_ms = pace_stride_ms
        if pace_budget_ms is not None:
            self.pace_budget_ms = pace_budget_ms
        self.checks = 0

    def check(self):
        self.checks += 1


class _FakeClock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


@pytest.fixture
def clock():
    c = _FakeClock(0.0)
    real = _pace._time.perf_counter
    _pace._time.perf_counter = c
    yield c
    _pace._time.perf_counter = real


# ── §4: the default (unpaced) path touches nothing this ask adds ─────────────────

def test_unpaced_cook_never_touches_the_cost_table(r):
    """The fifth assertion PACE-48-design.md §4 itself calls for, same shape as
    `test_fixpace_p2_gate_classification.py`'s existing four: an unpaced poll (`pace=False`,
    or a CPU device) must reach neither `_cost_lookup` nor `_cost_feed`, even when a caller
    passes a real `call_site_id`."""
    print("\n--- PACE-49: an unpaced cook never touches the cost table ---")
    calls = {"lookup": 0, "feed": 0}
    real_lookup, real_feed = _pace._cost_lookup, _pace._cost_feed

    def _counting_lookup(key):
        calls["lookup"] += 1
        return real_lookup(key)

    def _counting_feed(key, ms):
        calls["feed"] += 1
        return real_feed(key, ms)

    _pace._cost_lookup, _pace._cost_feed = _counting_lookup, _counting_feed
    try:
        with DeviceSpy():
            tok = _Token(pace=False)
            _pace.reset(tok, "cuda")
            for _ in range(5):
                _pace.paced_check(tok, "cuda", call_site_id="site-A")
            tok_cpu = _Token(pace=True)
            _pace.reset(tok_cpu, "cpu")
            for _ in range(5):
                _pace.paced_check(tok_cpu, "cpu", call_site_id="site-A")
    finally:
        _pace._cost_lookup, _pace._cost_feed = real_lookup, real_feed

    if calls == {"lookup": 0, "feed": 0}:
        r.ok("10 polls across pace=False and a CPU device touched the cost table 0 times")
    else:
        r.fail("PACE-49 unpaced gate", f"expected 0/0 lookup/feed calls, got {calls}")


def test_call_site_id_omitted_is_byte_identical_to_pre_pace49(r):
    """A caller that never passes `call_site_id` (every pre-PACE-49 call site) must see
    IDENTICAL pool bookkeeping to what `test_pace462_bounded_lookahead.py` already proves:
    a poll inside the stride window with the tail confirmed done still just skips, even
    with a tiny `pace_budget_ms` that would otherwise force a fall-through."""
    print("\n--- PACE-49: call_site_id=None is inert -- pre-existing behaviour byte-for-byte ---")
    with DeviceSpy() as spy, _clock_ctx() as clock:
        tok = _Token(pace=True, pace_depth=8, pace_stride_ms=10.0, pace_budget_ms=0.001)
        _pace.reset(tok, "cuda")
        _pace.paced_check(tok, "cuda")            # records (no call_site_id)
        clock.advance(0.001)
        _pace.paced_check(tok, "cuda")             # inside window, tail done -> should skip
        constructed = FakeCudaEvent._live
        outstanding = len(_pace._state.pool["outstanding"])  # noqa: SLF001
    if constructed == 1 and outstanding == 1:
        r.ok("an absurdly tiny budget never fires when call_site_id is never passed")
    else:
        r.fail("PACE-49 opt-out", f"constructed={constructed} outstanding={outstanding} "
               f"(expected 1/1 -- an omitted call_site_id must never force a record)")


import contextlib  # noqa: E402


@contextlib.contextmanager
def _clock_ctx():
    c = _FakeClock(0.0)
    real = _pace._time.perf_counter
    _pace._time.perf_counter = c
    try:
        yield c
    finally:
        _pace._time.perf_counter = real


# ── §6.1: the table updates only at a fresh peek-confirm or a wait, never a cache hit ──

def test_cost_table_feeds_only_on_a_fresh_confirmation_not_a_cache_hit(r):
    """Two economizing polls against the SAME already-confirmed tail (the PACE-47b cache
    path) must feed the table at most ONCE (the first, fresh confirmation) -- a cache hit
    answers from identity alone and must not re-attribute an interval already banked."""
    print("\n--- PACE-49: the cost table feeds on a fresh confirm, never a cache hit ---")
    feeds = []
    real_feed = _pace._cost_feed
    _pace._cost_feed = lambda key, ms: (feeds.append((key, ms)), real_feed(key, ms))
    try:
        with DeviceSpy(), _clock_ctx() as clock:
            tok = _Token(pace=True, pace_depth=8, pace_stride_ms=10.0)
            _pace.reset(tok, "cuda")
            _pace.paced_check(tok, "cuda", call_site_id="A")     # records E1, no tail yet
            clock.advance(0.001)
            _pace.paced_check(tok, "cuda", call_site_id="A")      # FRESH confirm of E1 (no
                                                                   # prior anchor -> no feed)
            clock.advance(0.001)
            _pace.paced_check(tok, "cuda", call_site_id="A")      # cache hit -> no feed
            clock.advance(0.001)
            _pace.paced_check(tok, "cuda", call_site_id="A")      # cache hit -> no feed
    finally:
        _pace._cost_feed = real_feed
    if feeds == []:
        r.ok("no prior timing anchor existed yet, so the first fresh confirm has nothing "
             "to attribute FROM -- and the two cache hits fed nothing either")
    else:
        r.fail("PACE-49 feed-on-fresh-only", f"expected 0 feeds, got {feeds}")


def test_cost_table_feeds_the_interval_between_two_real_records(r):
    """The real mechanism: poll 1 records E1 (call site A); the stride window elapses so
    poll 2 records E2 (call site B) for real; poll 3's fresh peek-confirm of E2 attributes
    the elapsed_time(E1, E2) interval to A (the call site active when E1 -- the anchor --
    was set) using FakeCudaEvent's own deterministic tick-based `elapsed_time`."""
    print("\n--- PACE-49: a real interval between two records feeds the FIRST call site's "
          "own entry ---")
    with DeviceSpy(), _clock_ctx() as clock:
        tok = _Token(pace=True, pace_depth=8, pace_stride_ms=10.0)
        _pace.reset(tok, "cuda")
        _pace.paced_check(tok, "cuda", call_site_id="A")   # records E1 (tick 0); anchor=None
                                                            # -> becomes (E1, "A")
        clock.advance(20.0)                                 # PAST the 10ms window
        _pace.paced_check(tok, "cuda", call_site_id="B")    # records E2 (tick 1) for real
                                                             # (stride elapsed) -- anchor was
                                                             # (E1, "A"); no fresh CONFIRM
                                                             # happened here (no peek on this
                                                             # path), so no feed yet either
        clock.advance(0.001)
        _pace.paced_check(tok, "cuda", call_site_id="B")    # inside window: fresh confirm
                                                             # of E2 -> feeds A with
                                                             # elapsed_time(E1, E2)
        est = _pace._cost_lookup(("A", _pace._state.device_idx,  # noqa: SLF001
                                   _pace._state.px_bucket))
    if est is not None and est[1] == 1 and abs(est[0] - FakeCudaEvent.TICK_MS) < 1e-9:
        r.ok(f"call site A's entry: ewma_ms={est[0]}, samples={est[1]} (one real tick "
             f"={FakeCudaEvent.TICK_MS}ms between E1 and E2)")
    else:
        r.fail("PACE-49 interval attribution", f"lookup('A', ...) = {est} (expected "
               f"(~{FakeCudaEvent.TICK_MS}, 1))")


# ── §6.2: cold-vs-warm -- the registry rule governs for exactly _WARMUP ticks ─────

def test_cold_call_site_never_forces_a_record_below_warmup(r):
    """A call site with samples < `_COST_WARMUP_SAMPLES` must never itself force a
    fall-through, no matter how a test rigs its (not-yet-trusted) estimate -- it rides the
    existing heavy/stride_s==0 rule, exactly PACE-47e's own behaviour. Simulated directly by
    feeding the table by hand (`_cost_feed`) fewer than `_COST_WARMUP_SAMPLES` times with a
    deliberately huge ms reading, then confirming a poll still economizes."""
    print("\n--- PACE-49: a not-yet-warm call site never forces a record ---")
    with DeviceSpy(), _clock_ctx() as clock:
        tok = _Token(pace=True, pace_depth=8, pace_stride_ms=10.0, pace_budget_ms=1.0)
        _pace.reset(tok, "cuda")
        key = ("loop", _pace._state.device_idx, _pace._state.px_bucket)  # noqa: SLF001
        for _ in range(_pace._COST_WARMUP_SAMPLES - 1):    # noqa: SLF001
            _pace._cost_feed(key, 9999.0)                  # huge -- would blow any budget
        _pace.paced_check(tok, "cuda", call_site_id="loop")   # records E1
        clock.advance(0.001)
        _pace.paced_check(tok, "cuda", call_site_id="loop")   # tail done; still COLD -> skip
        constructed = FakeCudaEvent._live
    if constructed == 1:
        r.ok(f"{_pace._COST_WARMUP_SAMPLES - 1} pre-fed huge samples (below warmup) never "  # noqa: SLF001
             f"forced a record -- only 1 event constructed")
    else:
        r.fail("PACE-49 cold-start", f"expected exactly 1 constructed event, got {constructed}")


def test_warm_call_site_over_budget_forces_a_fallthrough(r):
    """The fix's mechanism, mirroring PACE-47c's own repro shape: once a call site is WARM
    (>= `_COST_WARMUP_SAMPLES` real samples) with an estimate that alone exceeds
    `pace_budget_ms`, a poll must NOT honour the stride/peek skip even though the tail is
    confirmed done and `heavy=False` -- exactly the gap a proxy (footprint/pixel-threshold)
    cannot see, because this call site is neither halo-footprint nor above the pixel
    threshold."""
    print("\n--- PACE-49: a warm, over-budget call site forces the bound despite heavy=False ---")
    with DeviceSpy(), _clock_ctx() as clock:
        tok = _Token(pace=True, pace_depth=8, pace_stride_ms=10.0, pace_budget_ms=5.0)
        _pace.reset(tok, "cuda")
        key = ("loop", _pace._state.device_idx, _pace._state.px_bucket)  # noqa: SLF001
        for _ in range(_pace._COST_WARMUP_SAMPLES):        # noqa: SLF001
            _pace._cost_feed(key, 50.0)                     # far above the 5ms budget, warm
        _pace.paced_check(tok, "cuda", call_site_id="loop")    # records E1 (1/8)
        clock.advance(0.001)
        waits_before = sum(ev.sync_calls for ev in
                            list(_pace._state.pool["outstanding"]) + _pace._state.pool["free"])  # noqa: SLF001
        _pace.paced_check(tok, "cuda", call_site_id="loop", heavy=False)   # tail done, but
                                                                            # WARM + over
                                                                            # budget -> record
        outstanding_after = len(_pace._state.pool["outstanding"])  # noqa: SLF001
        constructed_after = FakeCudaEvent._live
    if outstanding_after == 2 and constructed_after == 2:
        r.ok("the warm, 50ms-estimated call site forced a real record on the second poll "
             "despite heavy=False and the tail being confirmed done")
    else:
        r.fail("PACE-49 budget fallthrough", f"outstanding_after={outstanding_after} "
               f"constructed_after={constructed_after} (expected 2/2)")


def test_within_budget_warm_call_site_still_economizes(r):
    """Regression: a WARM call site whose estimate comfortably fits under budget must still
    economize exactly as before -- the gate must not become "always record once warm"."""
    print("\n--- PACE-49: a warm, within-budget call site still economizes ---")
    with DeviceSpy(), _clock_ctx() as clock:
        tok = _Token(pace=True, pace_depth=8, pace_stride_ms=10.0, pace_budget_ms=5.0)
        _pace.reset(tok, "cuda")
        key = ("cheap", _pace._state.device_idx, _pace._state.px_bucket)  # noqa: SLF001
        for _ in range(_pace._COST_WARMUP_SAMPLES):        # noqa: SLF001
            _pace._cost_feed(key, 0.05)                      # tiny, well under 5ms
        _pace.paced_check(tok, "cuda", call_site_id="cheap")
        clock.advance(0.001)
        _pace.paced_check(tok, "cuda", call_site_id="cheap")
        constructed = FakeCudaEvent._live
    if constructed == 1:
        r.ok("a warm, cheap call site (0.05ms, budget 5ms) still economized")
    else:
        r.fail("PACE-49 within-budget regression", f"expected 1 constructed event, got "
               f"{constructed}")


def test_running_sum_accumulates_across_distinct_call_sites(r):
    """SUM semantics (PACE-48-design.md §2): several DISTINCT warm call sites, each alone
    under budget, must still force a fall-through once their SUM exceeds it -- a poll-count
    cap or a per-site-alone check would miss this."""
    print("\n--- PACE-49: several under-budget call sites still sum past the budget ---")
    with DeviceSpy(), _clock_ctx() as clock:
        tok = _Token(pace=True, pace_depth=8, pace_stride_ms=10.0, pace_budget_ms=5.0)
        _pace.reset(tok, "cuda")
        idx, bkt = _pace._state.device_idx, _pace._state.px_bucket  # noqa: SLF001
        for name in ("s1", "s2", "s3"):
            for _ in range(_pace._COST_WARMUP_SAMPLES):     # noqa: SLF001
                _pace._cost_feed((name, idx, bkt), 2.0)       # 2ms each, 3x2=6ms > 5ms budget
        _pace.paced_check(tok, "cuda", call_site_id="s1")     # records E1
        clock.advance(0.001)
        _pace.paced_check(tok, "cuda", call_site_id="s1")     # confirm E1; running=2.0<=5 -> skip
        _pace.paced_check(tok, "cuda", call_site_id="s2")     # cached tail; running=4.0<=5 -> skip
        outstanding_before_s3 = len(_pace._state.pool["outstanding"])  # noqa: SLF001
        _pace.paced_check(tok, "cuda", call_site_id="s3")     # running would be 6.0>5 -> record
        outstanding_after_s3 = len(_pace._state.pool["outstanding"])  # noqa: SLF001
    if outstanding_before_s3 == 1 and outstanding_after_s3 == 2:
        r.ok("s1 (2ms) then s2 (running 4ms) still skipped; s3 pushed the running sum to "
             "6ms > 5ms budget and forced a record")
    else:
        r.fail("PACE-49 running sum", f"before s3={outstanding_before_s3} (expected 1), "
               f"after s3={outstanding_after_s3} (expected 2)")


def test_pace_budget_ms_zero_disables_the_dimension(r):
    """The escape hatch, mirroring `pace_stride_ms=0`: a warm, wildly-over-any-sane-budget
    call site must never force a record when `pace_budget_ms=0`."""
    print("\n--- PACE-49: pace_budget_ms=0 disables the ms-budget dimension ---")
    with DeviceSpy(), _clock_ctx() as clock:
        tok = _Token(pace=True, pace_depth=8, pace_stride_ms=10.0, pace_budget_ms=0)
        _pace.reset(tok, "cuda")
        key = ("loop", _pace._state.device_idx, _pace._state.px_bucket)  # noqa: SLF001
        for _ in range(_pace._COST_WARMUP_SAMPLES):         # noqa: SLF001
            _pace._cost_feed(key, 9999.0)
        _pace.paced_check(tok, "cuda", call_site_id="loop")
        clock.advance(0.001)
        _pace.paced_check(tok, "cuda", call_site_id="loop")
        constructed = FakeCudaEvent._live
    if constructed == 1:
        r.ok("pace_budget_ms=0 economized despite a huge warm estimate")
    else:
        r.fail("PACE-49 budget escape hatch", f"expected 1 constructed event, got "
               f"{constructed}")


def test_pace_budget_ms_default_when_absent(r):
    print("\n--- PACE-49: pace_budget_ms absent resolves to the module default ---")
    tok = _Token(pace=True, pace_budget_ms=None)
    tok.__dict__.pop("pace_budget_ms", None)
    resolved = _pace._resolve_budget_ms(tok)  # noqa: SLF001
    if resolved == _pace._DEFAULT_BUDGET_MS:  # noqa: SLF001
        r.ok(f"resolved default budget {resolved}ms")
    else:
        r.fail("PACE-49 default budget", f"expected {_pace._DEFAULT_BUDGET_MS}, got {resolved}")


@pytest.mark.parametrize("bad_budget", [-1, -0.5, "3", True, False])
def test_pace_budget_ms_rejects_invalid(bad_budget):
    tok = _Token(pace=True)
    tok.pace_budget_ms = bad_budget
    with pytest.raises(ValueError):
        _pace._resolve_budget_ms(tok)  # noqa: SLF001


@pytest.mark.parametrize("bad_budget", [float("nan"), float("inf"), float("-inf")])
def test_pace_budget_ms_rejects_nonfinite(bad_budget):
    tok = _Token(pace=True)
    tok.pace_budget_ms = bad_budget
    with pytest.raises(ValueError):
        _pace._resolve_budget_ms(tok)  # noqa: SLF001


# ── §5: scale/ROI inheritance -- a different px_bucket is a DIFFERENT key ────────

def test_scaled_down_cook_gets_an_independent_estimate_from_full_canvas(r):
    """PACE-48-design.md §5: a scaled-down or ROI-windowed cook's own call-site estimate
    must differ from (be independent of) the same call site's full-canvas estimate --
    proving inheritance via the shared `px_bucket` key, not merely asserting it. Two
    `reset()`s at different `spatial_shape`s must resolve DIFFERENT `px_bucket`s, and
    `_cost_lookup` for one bucket must never see the other's samples."""
    print("\n--- PACE-49: a scaled/ROI-windowed cook keys an independent cost-table entry ---")
    with DeviceSpy():
        tok_small = _Token(pace=True)
        _pace.reset(tok_small, "cuda", spatial_shape=(1, 64, 64))
        small_bucket = _pace._state.px_bucket  # noqa: SLF001
        small_idx = _pace._state.device_idx  # noqa: SLF001
        _pace._cost_feed(("stmt", small_idx, small_bucket), 1.0)

        tok_big = _Token(pace=True)
        _pace.reset(tok_big, "cuda", spatial_shape=(1, 4096, 4096))
        big_bucket = _pace._state.px_bucket  # noqa: SLF001
        big_idx = _pace._state.device_idx  # noqa: SLF001
        big_lookup = _pace._cost_lookup(("stmt", big_idx, big_bucket))

    if small_bucket != big_bucket and big_lookup is None:
        r.ok(f"64^2 -> bucket {small_bucket}, 4096^2 -> bucket {big_bucket} (distinct); "
             f"the small cook's sample never leaked into the big bucket's lookup")
    else:
        r.fail("PACE-49 bucket independence", f"small_bucket={small_bucket} "
               f"big_bucket={big_bucket} (must differ), big_lookup={big_lookup} "
               f"(expected None)")


# ── the bounded table never grows past _COST_TABLE_MAX ───────────────────────────

def test_cost_table_is_bounded(r):
    print("\n--- PACE-49: the cost table is a bounded LRU ---")
    for i in range(_pace._COST_TABLE_MAX + 50):  # noqa: SLF001
        _pace._cost_feed((f"site-{i}", 0, 1), 1.0)
    size = len(_pace._COST_TABLE)  # noqa: SLF001
    if size == _pace._COST_TABLE_MAX:  # noqa: SLF001
        r.ok(f"table holds exactly {size} entries after {_pace._COST_TABLE_MAX + 50} feeds")  # noqa: SLF001
    else:
        r.fail("PACE-49 bounded table", f"expected {_pace._COST_TABLE_MAX}, got {size}")  # noqa: SLF001
