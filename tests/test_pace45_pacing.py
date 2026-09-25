"""
PACE-45(b) — paced cancellation: a v0.45.2 patch for a host-reported REGRESSION. On CUDA,
a cancel token no longer bounds a GPU-heavy cook.

**The bug.** Since the v0.41-v0.44 readback removals (TRK-66/67/68) and v0.44's stencil-
route work, nothing paces the host thread to the device any more. Every cancel-poll point
(the interpreter's per-top-level-statement poll, the cancel-aware codegen tier's in-body
`_CK()` polls, the stencil route's entry poll, and a naturally multi-pass builtin's
`poll_cook_cancel` between its own passes) only ever fires while the host is still QUEUEING
kernels — on CUDA that takes milliseconds even for a long chain of heavy statements, so a
token that trips once the real GPU work is under way has no yield point left to reach until
the whole queue drains. A host's untrusted-tool time ceiling, or any other cancel that is
meant to bound REAL work, no longer bounds anything on CUDA.

**The fix (`tex_runtime/pacing.py`).** Opt-in, via a truthy `pace` attribute on the token
(`pacing.wants_pacing`): a paced poll point records a CUDA event, then waits for the event
recorded at the PREVIOUS poll point before letting the caller queue further work — bounding
the host to at most one poll-interval of device time ahead — polling the token in a short
`event.query()` loop (never a blocking `synchronize()`) so a trip mid-wait raises promptly.
`cancel=None`, a CPU cook, or a token with no `pace` attribute (every caller before this ask)
takes the exact `token.check()` path `host._cancel_check` always ran — see
`test_pace45_wants_pacing_and_unpaced_check` for the deterministic, non-GPU proof of that.

**Why opt-in, not default-on-whenever-a-token-is-passed.** A host's own cancel token rides
EVERY cook, interactive included (SCHED-3's whole point) — so turning pacing on for any
non-None token would move the default path's timing for every existing caller, and
measurement (see this ask's hand-back) shows that cost is not negligible: pacing removes the
overlap between a statement's CPU-side dispatch and the PREVIOUS statement's still-running
GPU work, which is exactly the overlap invariant 7's default path already relies on.

The heavy repro tests below need a real CUDA device — there is no CPU witness for "the host
queued ahead of the device," since a CPU cook has no such queue at all. They are two of the
SIMP-3 skip sites this ask adds; see `tests/test_simp3_skip_budget.py`'s re-pin note.
`CookResult.done` (the other half of this ask, Q3) has its own file, `test_pace45_done_event.py`.
"""
import threading

import pytest

from helpers import *  # noqa: F401,F403  (SubTestResult, torch, time, make_img)
from TEX_Wrangle import tex_engine
from TEX_Wrangle.tex_runtime.host import CookCancelled
from TEX_Wrangle.tex_runtime import pacing as _pace


# ── The opt-in gate itself, deterministic, no GPU timing ────────────────────

def test_pace45_wants_pacing_and_unpaced_check(r: SubTestResult):
    """Unit-level proof of the opt-in contract, independent of any GPU timing: a token with
    no `pace` attribute (every token before this ask, including ComfyUI's own interrupt
    bridge) reads as NOT wanting pacing, and `paced_check` on it does exactly what
    `host._cancel_check` always did -- one `token.check()` call, nothing more, regardless of
    device. A token that sets `pace=True` reads as wanting it."""
    class _Bare:
        def __init__(self):
            self.calls = 0

        def check(self):
            self.calls += 1

    tok = _Bare()
    if _pace.wants_pacing(tok):
        r.fail("PACE-45 opt-in gate", "a bare token (no `pace` attribute) reads as wanting pacing")
    else:
        r.ok("a bare token (no `pace` attribute) reads as NOT wanting pacing")

    _pace.paced_check(tok, "cpu")
    if tok.calls == 1:
        r.ok("a bare token's paced_check is exactly one token.check() call")
    else:
        r.fail("PACE-45 opt-in gate", f"expected exactly 1 check() call, got {tok.calls}")

    tok.pace = True
    if _pace.wants_pacing(tok):
        r.ok("the same token, once `pace = True` is set, reads as wanting pacing")
    else:
        r.fail("PACE-45 opt-in gate", "a token with pace=True does not read as wanting pacing")

    # cancel=None must never even reach wants_pacing/_is_cuda -- paced_check(None, ...) is a
    # bare no-op, exactly _cancel_check(None)'s own body.
    _pace.paced_check(None, "cuda")
    r.ok("paced_check(None, ...) is a no-op (never touches the token or the device)")


def test_pace45_unpaced_bit_exact_cpu(r: SubTestResult):
    """Zero-drift check on CPU (always available, no skip needed): cancel=None, a bare
    token, and a token that explicitly opts in (`pace=True`, live/never-tripping) must all
    cook byte-identical pixels -- PACE-45's mechanism is a CUDA-only branch and must never be
    reachable, or change a single bit, off CUDA."""
    class _Live:
        def __init__(self, pace=False):
            self.pace = pace

        def check(self):
            pass

    code = "@OUT = gauss_blur(@A, 5.0);"
    img = make_img(1, 40, 40, 4, seed=451)
    base = tex_engine.cook(code, {"A": img.clone()}, device_mode="cpu")
    plain = tex_engine.cook(code, {"A": img.clone()}, device_mode="cpu", cancel=_Live(pace=False))
    paced = tex_engine.cook(code, {"A": img.clone()}, device_mode="cpu", cancel=_Live(pace=True))
    md1 = (base.outputs["OUT"].float() - plain.outputs["OUT"].float()).abs().max().item()
    md2 = (base.outputs["OUT"].float() - paced.outputs["OUT"].float()).abs().max().item()
    if md1 == 0.0 and md2 == 0.0:
        r.ok("CPU cooks are bit-exact regardless of a live token's `pace` attribute")
    else:
        r.fail("PACE-45 CPU bit-exactness", f"maxdiff plain={md1} paced={md2}")


# ── The heavy CUDA repro: 20 sequential gauss_blur statements at 2560^2 ─────
#
# Scaled down from the finding's 64 statements at 8K (which drains for 5.5-8.5s on the box
# that found this) to keep this row runnable in the ordinary suite rather than only under
# `slow`/`timing`: the MECHANISM under test (a poll point that waits on the device between
# statements) does not care about resolution or statement count, only that queueing is fast
# relative to real GPU work, which this shape still is (measured on this box: ~75ms to queue
# vs a ~275ms synced completion for the SAME uncancelled program).

_N_STATEMENTS = 20
_SIZE = 2560
_PROGRAM = "vec4 x = @A;\n" + "x = gauss_blur(x, 8.0);\n" * _N_STATEMENTS + "@OUT = x;\n"


def _pace45_bindings(seed: int):
    return {"A": make_img(1, _SIZE, _SIZE, 4, seed=seed).cuda()}


class _LiveToken:
    """A CancelToken (PACE-45) that never trips. `pace=True` opts into bounded-queue-ahead
    pacing."""
    def __init__(self, pace: bool = True):
        self.pace = pace

    def check(self) -> None:
        pass


class _ThreadTripToken:
    """Trips from a BACKGROUND thread after `delay_s` seconds -- the shape the finding
    measured: an interrupt arriving asynchronously mid-cook, never from the cooking thread
    itself."""
    def __init__(self, delay_s: float, pace: bool = True):
        self.pace = pace
        self._tripped = threading.Event()
        self._timer = threading.Timer(delay_s, self._tripped.set)
        self._timer.daemon = True
        self._timer.start()

    def check(self) -> None:
        if self._tripped.is_set():
            raise CookCancelled("PACE-45 repro: deadline token tripped")


def _measure_full_runtime_cuda() -> float:
    """The TRUE uncancelled GPU completion time (drain included), independent of pacing --
    an explicit `torch.cuda.synchronize()` after `cook()` returns, mirroring how the finding
    itself measured "the GPU drained for about 5.5-8.5s". Discards a cold-cache first leg
    (docs/brief-conventions.md's measurement rule) and takes the FLOOR of two warm legs, the
    same 'lower-bound scale, not a precise duration' floor `test_v044_cancel44.py` uses. A
    `synchronize()` between legs keeps one leg's tail from backing up into the next's queue."""
    def once(seed):
        t0 = time.perf_counter()
        tex_engine.cook(_PROGRAM, _pace45_bindings(seed), device_mode="cuda")
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    once(900)  # discard cold leg
    return min(once(901), once(902))


def test_pace45_cuda_pacing_bit_exact_and_repro(r: SubTestResult):
    """The core PACE-45(b) repro (non-timing half), plus the bit-exactness check it shares a
    CUDA guard with. Both need a real CUDA device; there is no CPU witness for either."""
    if not torch.cuda.is_available():
        r.skip("PACE-45 CUDA repro", "no CUDA on this box")
        return

    full = _measure_full_runtime_cuda()

    # -- No pixel change: a paced, uncancelled cook is bit-exact with the plain route --
    plain = tex_engine.cook(_PROGRAM, _pace45_bindings(910), device_mode="cuda")
    torch.cuda.synchronize()
    paced = tex_engine.cook(_PROGRAM, _pace45_bindings(910), device_mode="cuda",
                            cancel=_LiveToken(pace=True))
    torch.cuda.synchronize()
    md = (plain.outputs["OUT"].float() - paced.outputs["OUT"].float()).abs().max().item()
    if md == 0.0:
        r.ok("a paced, uncancelled CUDA cook is bit-exact with the plain (cancel=None) route")
    else:
        r.fail("PACE-45 CUDA bit-exactness", f"maxdiff {md}")

    # -- The repro itself: a token armed for pacing, tripped from another thread at 25% of
    #    the uncancelled runtime, must raise well before that runtime elapses. At v0.45.1
    #    this never raised at all (the whole queue -- 64 statements at 8K there -- was
    #    already launched in ~15ms, long before a mid-cook trip had anything left to reach).
    delay = full * 0.25
    tok = _ThreadTripToken(delay, pace=True)
    t0 = time.perf_counter()
    try:
        tex_engine.cook(_PROGRAM, _pace45_bindings(920), device_mode="cuda", cancel=tok)
        r.fail("PACE-45 repro", f"cook completed without raising (full={full * 1000:.0f}ms, "
               f"trip delay={delay * 1000:.0f}ms) -- the regression is back")
        return
    except CookCancelled:
        elapsed = time.perf_counter() - t0
    torch.cuda.synchronize()

    if elapsed < full:
        r.ok(f"paced cancel raised after {elapsed * 1000:.0f}ms, well inside the "
             f"{full * 1000:.0f}ms uncancelled runtime (trip armed at {delay * 1000:.0f}ms)")
    else:
        r.fail("PACE-45 repro", f"raised, but only after {elapsed * 1000:.0f}ms -- not before "
               f"the {full * 1000:.0f}ms uncancelled runtime")


@pytest.mark.timing
def test_pace45_cuda_repro_latency(r: SubTestResult):
    """Timing half of the same repro: the bound is against the TRIP, not the whole runtime --
    how promptly the cancel actually lands once pacing is armed. CUDA, quiet-box
    informational, deselected by default (`-m 'not timing'`); run alone with `-m timing`."""
    if not torch.cuda.is_available():
        r.skip("PACE-45 CUDA repro latency", "no CUDA on this box")
        return

    full = _measure_full_runtime_cuda()
    delay = full * 0.25
    trials = 5
    latencies = []
    for i in range(trials):
        tok = _ThreadTripToken(delay, pace=True)
        t0 = time.perf_counter()
        try:
            tex_engine.cook(_PROGRAM, _pace45_bindings(930 + i), device_mode="cuda", cancel=tok)
            r.fail("PACE-45 repro latency", "cook completed without raising")
            return
        except CookCancelled:
            latencies.append(time.perf_counter() - t0 - delay)
        torch.cuda.synchronize()

    worst = max(latencies)
    # Generous, box-robust: the wait between poll points is bounded by roughly one
    # poll-interval of GPU work (this program's per-statement share of `full`), not by the
    # whole remaining queue -- half the full runtime is a wide margin over that bound.
    bound = full * 0.5
    if worst < bound:
        r.ok(f"cancel landed {worst * 1000:.1f}ms after the trip (worst of {trials}), under "
             f"the {bound * 1000:.0f}ms bound (full runtime {full * 1000:.0f}ms)")
    else:
        r.fail("PACE-45 repro latency", f"worst-case landing {worst * 1000:.1f}ms exceeds the "
               f"{bound * 1000:.0f}ms bound")
