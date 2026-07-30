"""
SCHED-4 — the two-tier preemptive cook queue.

The engine could already be *stopped*: SCHED-3 threads a `CancelToken` through every cook,
so a host's Stop button reaches a running tier, a tiled strip loop, and the interpreter's
statement walk. This module is the generalization the compositor reports ask for — stop for
a reason other than the user giving up: **something more important arrived.**

Design note: `docs/cook-queue-scheduling.md`. Two findings from it are load-bearing here and
are repeated so this file reads alone:

  * **The preemption points already exist.** SCHED-3's yield sites are per tier attempt, per
    tile/batch/halo strip, and per TOP-LEVEL interpreter statement. `compile_fused` splices a
    chain into ONE program whose statements carry `loc.stage` (Q-4), so every fused-stage
    boundary is already a poll site. SCHED-4 adds no instrumentation to the cook path: it
    mints the token, decides who may trip it, and decides what happens to the job that was
    tripped. That is the whole item.
  * **One worker thread, not a pool.** The tree-walking interpreter is Python-heavy and holds
    the GIL between kernels; a concurrent Tier-B cook would not overlap Tier A, it would
    interleave two Python interpreters onto one core and slow Tier A down — the one outcome
    this item exists to prevent. Mirrors `compiled._COMPILE_POOL`'s max_workers=1 discipline.
    GRAPH-2's parallel region executor is what reopens it.

THREE admission classes, not two (doc 39 is explicit, and the reason is a real workflow — a
user starts a render and keeps working):

    INTERACTIVE   preempts everything below; never preempted; never shed.
    COMMITTED     a render the user explicitly started. Pauses under INTERACTIVE; NEVER shed.
    SPECULATIVE   predictive/idle work. Pauses under both; shed FIRST under pressure.

Two classes cannot express that: COMMITTED and SPECULATIVE are identical in priority (both
pause for Tier A) and opposite in durability (one is never dropped, the other is dropped
first). PRED-1 (v0.31, same release) adds the confidence/reason surface and the shed policy
on top of the SPECULATIVE class; this module owns the queue, the preemption, and the re-queue.

PREEMPT vs SHED are deliberately distinct outcomes, because collapsing them into one
`cancelled` flag is how a scheduler silently loses work:

    preempt → the job returns to the HEAD of its class. Transient; the host is never told.
    shed    → the job finishes CANCELLED and its waiter wakes. Terminal.

THE CORRECTNESS HEART (doc §4b): *the preempt flag is consulted only by the token, never by
the worker.* Preemption is cooperative — the flag is set and the cook notices it at its next
yield point. If the cook RETURNS first, the result is real, correct, and already paid for, so
the job completes with that value whatever flags were raised while it ran. Written the other
way round (worker checks the flag after the cook and discards) the scheduler throws away
completed frames under exactly the load where frames are most expensive.

INVARIANT #7: nothing on the ComfyUI cook path imports or reaches this module. No thread
exists until a host constructs a `CookQueue`, and none starts until the first `submit()`.
The queue passes its token through the existing `cancel=` parameter, so from `tex_engine`'s
side a queued cook is indistinguishable from a SCHED-3 cook with a Stop button attached —
the diff to the cook path is empty, which is what makes the invariant provable, not asserted.
"""
from __future__ import annotations

import itertools
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

from .tex_runtime import profile as _profile
from .tex_runtime.host import CookCancelled

logger = logging.getLogger("TEX")


# ── Admission classes ────────────────────────────────────────────────────────
# Ordered by priority: LOWER RANKS FIRST, so a plain `<` is "outranks" and the three deques
# can be scanned in numeric order. Exposed as ints (not an Enum) for the same reason autotier
# exposes plain strings — a test, a host, and a JSON stats blob all speak them without imports.

INTERACTIVE = 0
COMMITTED = 1
SPECULATIVE = 2

CLASSES = (INTERACTIVE, COMMITTED, SPECULATIVE)
CLASS_NAMES = {INTERACTIVE: "interactive", COMMITTED: "committed", SPECULATIVE: "speculative"}

#: Classes that may be shed under pressure. COMMITTED is absent BY CONTRACT — the user asked
#: for that render by name, and dropping it to make room for a prefetch is a bug they
#: experience as "my render vanished".
SHEDDABLE = frozenset({SPECULATIVE})

# Job states. Terminal states are DONE / FAILED / CANCELLED.
PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"
#: IO-1: submitted, but one or more of its `inputs=` promises has not landed. NOT eligible —
#: the worker never sees it, which is what keeps the single-worker contract intact while a
#: branch waits on host I/O. A landing callback flips it to PENDING.
WAITING = "waiting"

_TERMINAL = frozenset({DONE, FAILED, CANCELLED})


class _Preempted(CookCancelled):
    """Raised by `_JobToken` and by nothing else — the PROVENANCE of a cancellation, carried on
    the signal instead of on a flag.

    The worker has to answer one question when a cook aborts: *did the queue itself ask for
    this?* Only then is the abort transient and the job re-queueable. Inferring that from a
    `preempt_requested` flag is a livelock in two distinct ways, both measured: not checking at
    all requeues a permanently-tripped host token forever (452,729 requeues/second), and checking
    the REQUEST rather than who raised misclassifies a foreign cancel that lands while a preempt
    is merely outstanding. A subclass makes the two cases structurally exclusive — `except
    _Preempted` before `except CookCancelled` — instead of flag-and-guard exclusive, so the bug
    is unrepresentable rather than defended against."""


class _JobToken:
    """The SCHED-3 `CancelToken` the queue mints for ONE run attempt of one job.

    A fresh token per attempt, not per job: a re-queued job's previous token has already
    tripped, and reusing it would abort the retry at its first yield point — an infinite
    preempt/requeue loop that looks exactly like a hang.

    `check()` sits at every SCHED-3 yield point, so it must stay a few attribute loads and two
    branches. Shed is tested first: a job that is both shed and preempted is being torn down,
    and the terminal outcome must win over the transient one.

    Only this token raises `_Preempted`, which is how the worker knows a cancellation was the
    queue's own doing and therefore transient — see that class.

    THE QUANTUM IS ENFORCED HERE, not at the submit site, and that is the difference between
    DEFERRING a preemption and DROPPING one. Refusing to set the flag while a job is inside its
    quantum loses the request outright — nothing re-attempts it, so an interactive frame waits
    for the whole running cook. Setting the flag immediately and honouring it at the first yield
    point *after* the quantum bounds Tier A's wait to `quantum + one yield` with no timer, no
    re-attempt bookkeeping, and no lost request."""
    __slots__ = ("_job", "_quantum_ms")

    def __init__(self, job: "Job", quantum_ms: float = 0.0):
        self._job = job
        self._quantum_ms = quantum_ms

    def check(self) -> None:
        j = self._job
        if j.shed_requested:
            raise CookCancelled(f"cook shed ({CLASS_NAMES.get(j.klass, j.klass)}): {j.reason or 'no reason given'}")
        if j.preempt_requested:
            if self._quantum_ms and \
                    (time.perf_counter() - j.started_at) * 1000.0 < self._quantum_ms:
                return                   # inside its quantum: the request stands, deferred
            raise _Preempted("cook preempted by a higher-priority request")


@dataclass(eq=False)
class Job:
    """One submitted cook. Mutable — the worker and the submitter both write it under the
    queue's lock, and the handle a host holds is a view of this object.

    `eq=False` is load-bearing, not style: the queue removes a job from its deque with
    `deque.remove`, which searches by `==`. A generated `__eq__` compares every field, so two
    jobs submitted with the same callable and class would be interchangeable and `cancel()`
    could remove the wrong one. Identity is what a job is.

    `fn` takes exactly one argument, the token, and is expected to hand it to
    `tex_engine.cook(cancel=...)`. That signature is the whole coupling between the queue and
    the engine: the queue never imports `tex_engine`."""
    id: int
    klass: int
    fn: Callable[[Any], Any]
    reason: str = ""
    #: PRED-1's surface. `confidence` is the host's P(this work is wanted); `profile_key`/`px`
    #: let the policy ask PROF-1 what it will cost; `cost_ms` overrides that lookup (a test
    #: feeding explicit numbers, or a host with its own estimate). `score` orders the
    #: SPECULATIVE queue and picks the shed victim.
    confidence: float = 1.0
    profile_key: tuple | None = None
    px: int | None = None
    cost_ms: float | None = None
    score: float = 0.0
    #: IO-1's PROF-1 pollution guard. An I/O-bound job (a DATA-7 prefetch) must NOT feed the
    #: compute cost table: an I/O wait recorded as compute cost poisons CACHE-7 placement and
    #: PRED-1 admission in one stroke. Passing no `profile_key` already skips the feedback —
    #: this exists because that is an accident of what the caller passed, and the caller is
    #: the host. An explicit flag is a guard; "we happened not to pass a key" is not.
    feeds_profile: bool = True
    #: IO-1: promises this job's cook needs. Duck-typed — anything with `.landed`, `.error`
    #: and `.on_land(cb)` works, so the queue never imports the marshalling seam and knows
    #: nothing about bindings. `tex_marshalling.Promise` is the shipped implementation.
    inputs: tuple = ()

    state: str = PENDING
    value: Any = None
    error: BaseException | None = None
    preempt_requested: bool = False
    shed_requested: bool = False
    #: Set when this job was head-requeued after a preemption, cleared when it next starts.
    #: It marks the FIFO resume prefix that sits ahead of the score-ordered region — see
    #: `_enqueue_locked`.
    resumed: bool = False
    attempts: int = 0
    #: How many times this job has been preempted. The starvation brake reads it: a job that has
    #: used its budget is no longer preemptible, so it finishes instead of being retried forever.
    preemptions: int = 0
    started_at: float = 0.0        # set per attempt; the PROF-1 feedback measures from it
    _done: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def class_name(self) -> str:
        return CLASS_NAMES.get(self.klass, str(self.klass))

    # ── the host-facing handle surface (a Job IS its own handle) ──
    def wait(self, timeout: float | None = None) -> bool:
        """Block until this job reaches a terminal state. True if it did, False on timeout."""
        return self._done.wait(timeout)

    def result(self, timeout: float | None = None):
        """The cook's return value. Raises whatever the cook raised, `CookCancelled` if the
        job was shed or cancelled, or `TimeoutError` if it has not finished in time."""
        if not self._done.wait(timeout):
            raise TimeoutError(f"job {self.id} ({self.class_name}) did not finish in {timeout}s")
        if self.state == DONE:
            return self.value
        if self.error is not None:
            raise self.error
        raise CookCancelled(f"job {self.id} was cancelled")


@dataclass
class QueueStats:
    """Counters a test and a host HUD both read. Deliberately plain ints — the queue's own
    behaviour has to be assertable without instrumenting it further."""
    submitted: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0          # shed, or cancelled by the host
    preempted: int = 0          # times a running job was told to yield
    requeued: int = 0           # times a preempted job went back on the queue
    refused: int = 0            # PRED-1 admission said no at submit time
    shed: int = 0               # dropped under pressure (a subset of `cancelled`)
    #: The starvation brake REFUSED a preemption because the job had spent its
    #: `max_preemptions` budget. The quantum is NOT counted here: it defers a preemption
    #: rather than refusing one (`_JobToken.check` honours the request at the first yield
    #: past the quantum), and counting a deferral as a refusal would over-report the brake.
    preempt_denied: int = 0
    #: IO-1: jobs currently parked on an unlanded input. Decremented when they wake or fail,
    #: so a non-zero value at rest means a host promised something it never delivered — the
    #: one symptom of a stuck pipeline that looks exactly like an idle queue.
    waiting: int = 0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class CookQueue:
    """A priority cook queue with cooperative preemption and a single worker thread.

    Construct one per host (or per session). Nothing constructs one for you — see the
    invariant-#7 note in the module docstring.

        q = CookQueue()
        job = q.submit(lambda cancel: tex_engine.cook(code, binds, cancel=cancel),
                       klass=INTERACTIVE)
        frame = job.result(timeout=5.0)

    Ordering is FIFO within a class. A PREEMPTED job goes to the HEAD of its class rather than
    the tail: it has already started, so finishing it is closer than starting a peer, and
    tail-requeue under sustained Tier-A pressure is a livelock (the same job yields forever
    while its peers cycle)."""

    def __init__(self, *, name: str = "tex-cook", min_quantum_ms: float = 15.0,
                 max_preemptions: int = 3):
        self._name = name
        #: The starvation brake — see `_preempt_for_locked`. A running job is un-preemptible for
        #: its first `min_quantum_ms`, and permanently un-preemptible once it has been preempted
        #: `max_preemptions` times. Without both, a render longer than the gap between
        #: interactive requests never completes (measured: 104 attempts, 0 completions).
        self.min_quantum_ms = float(min_quantum_ms)
        self.max_preemptions = int(max_preemptions)
        #: ONE lock, reached only as `self._wake` (a Condition is a context manager for its own
        #: lock). A second name for it would imply a two-lock discipline that does not exist.
        self._wake = threading.Condition(threading.Lock())
        self._q: dict[int, deque] = {k: deque() for k in CLASSES}
        self._running: Job | None = None
        self._closed = False
        self._ids = itertools.count(1)
        self._worker: threading.Thread | None = None
        self.stats = QueueStats()
        #: PRED-1 installs an admission policy here; None means "admit everything", which is
        #: SCHED-4's own behaviour and what the queue is tested against on its own.
        self._policy = None

    # ── submission ───────────────────────────────────────────────────────────
    def submit(self, fn: Callable[[Any], Any], *, klass: int = INTERACTIVE,
               reason: str = "", confidence: float = 1.0,
               profile_key: tuple | None = None, px: int | None = None,
               cost_ms: float | None = None, feeds_profile: bool = True,
               inputs=()) -> Job:
        """Queue `fn(cancel_token)` at `klass`. Returns the Job immediately (never blocks).

        A submit at a class that OUTRANKS the running job preempts it — the running job's
        token is tripped and it will yield at its next SCHED-3 yield point, then go back on
        the queue. Submitting at the same class as (or below) the running job never preempts:
        same-class work is FIFO, which is what makes a burst of slider frames orderly rather
        than a stampede of mutual aborts.

        `reason` / `confidence` / `cost_ms` / `profile_key` / `px` are PRED-1's surface, and
        inert unless a `SpeculativePolicy` is installed (see `install_policy`).
        `feeds_profile=False` (IO-1) excludes an I/O-bound job from PROF-1's cost table.
        `inputs=` (IO-1) is a list of promises: the job sits WAITING — enqueued but not
        eligible — until every one has landed, so the single worker never blocks on I/O."""
        if klass not in self._q:
            raise ValueError(f"unknown cook class {klass!r} (expected one of {CLASSES})")
        job = Job(id=next(self._ids), klass=klass, fn=fn, reason=reason,
                  confidence=float(confidence), profile_key=profile_key, px=px,
                  cost_ms=cost_ms, feeds_profile=bool(feeds_profile),
                  inputs=tuple(inputs or ()))
        with self._wake:
            if self._closed:
                raise RuntimeError("CookQueue is closed")
            if not self._admit_locked(job):
                self.stats.refused += 1
                self._finish_locked(job, CANCELLED, error=CookCancelled(
                    f"speculative cook refused by admission policy: {reason or 'no reason given'}"))
                return job
            self.stats.submitted += 1
            waiting = bool(job.inputs) and not self._inputs_ready(job)
            if waiting:
                job.state = WAITING
                self.stats.waiting += 1
            self._enqueue_locked(job)
            # A WAITING job must NOT preempt. Tripping a running render for a job that cannot
            # start is the worst trade available: the render loses all its progress (§4a —
            # there is no resume) and the preemptor is still waiting on a disk read.
            if not waiting:
                self._preempt_for_locked(klass)
            try:
                self._shed_locked()
            except BaseException:            # noqa: BLE001 — a host policy bug is not ours
                # The job is already enqueued here, so letting this escape would fail
                # `submit()` for the caller while the cook still runs on the next wakeup —
                # side effects and all, with no handle to observe them through.
                logger.exception("[TEX] cook-queue shed policy raised; ignoring")
            self._wake.notify_all()
        # Registered OUTSIDE the queue's lock. `on_land` fires an already-landed promise's
        # callback synchronously, and that callback takes this lock — arming under it would
        # deadlock on the exact race the WAITING state is for (a promise that lands between
        # the readiness check above and here).
        for p in job.inputs:
            try:
                p.on_land(self._on_input_landed)
            except Exception:
                logger.exception("[TEX] promise refused a landing callback; waking the job")
                self._wake_if_ready(job)
        if self._worker is None:      # benign unlocked double-check; _ensure_worker re-tests
            self._ensure_worker()
        return job

    @staticmethod
    def _inputs_ready(job: Job) -> bool:
        return all(getattr(p, "landed", True) for p in job.inputs)

    def _on_input_landed(self, _promise) -> None:
        """A promise landed: wake every WAITING job whose inputs are now all in.

        Scans rather than indexing promise -> jobs. The waiting set is bounded by the
        queue's own depth and a landing is a once-per-frame event, so an index would be a
        second structure to keep consistent with the deques for no measurable gain."""
        with self._wake:
            for dq in self._q.values():
                for job in list(dq):
                    if job.state == WAITING:
                        self._wake_if_ready_locked(job)
            self._wake.notify_all()

    def _wake_if_ready(self, job: Job) -> None:
        with self._wake:
            self._wake_if_ready_locked(job)
            self._wake.notify_all()

    def _wake_if_ready_locked(self, job: Job) -> None:
        """WAITING -> PENDING once every input has landed, or -> FAILED if one failed.

        A failed promise fails its jobs rather than leaving them parked: a source that
        cannot be read is a finished question, not a slow one, and a job waiting forever on
        it is indistinguishable from a hung queue."""
        if job.state != WAITING:
            return
        # P0-B: a job the host cancelled is NOT woken by a later landing. Without this the
        # landing granted a preemption on behalf of a job that can never run.
        #
        # DEFENSIVE, and deliberately kept as such: every path that sets `shed_requested`
        # (`cancel`, `_shed_speculative_locked`) also removes the job from its deque under
        # this same lock, so a WAITING+shed job in a deque is unreachable today and a
        # mutation row for this branch is unkillable. It stays because the removal and the
        # flag are set by DIFFERENT call sites, and one future path that sets the flag
        # without removing would resurrect the whole defect silently.
        if job.shed_requested:
            self._remove_locked(job)
            self.stats.cancelled += 1
            self._finish_locked(job, CANCELLED,
                                error=CookCancelled(f"job {job.id} was cancelled while waiting"))
            return
        for p in job.inputs:
            err = getattr(p, "error", None)
            if err is not None:
                self._remove_locked(job)
                # P0-F: the class-dependent host-I/O rule lived only in `_run_one`, so a
                # promise that FAILED before its job ever ran alarmed even when the job was
                # SPECULATIVE — FAILED state, a raw E7xxx handed to the waiter, and no
                # refusals-ledger row. Speculation never alarms, whichever door the failure
                # arrives through.
                if job.klass == SPECULATIVE and str(getattr(err, "_code", "")).startswith("E7"):
                    self._note_speculative_io_failure(job, err)
                    self.stats.cancelled += 1
                    self._finish_locked(job, CANCELLED, error=CookCancelled(
                        f"speculative host-I/O failure ({getattr(err, '_code', 'E7')}): {err}"))
                    return
                self.stats.failed += 1
                self._finish_locked(job, FAILED, error=err)   # decrements `waiting` itself
                return
        if not self._inputs_ready(job):
            return
        self.stats.waiting -= 1
        job.state = PENDING
        # Now that it can actually run, it gets the preemption it was denied at submit.
        self._preempt_for_locked(job.klass)

    def _enqueue_locked(self, job: Job, *, head: bool = False) -> None:
        """Place a PENDING job. SPECULATIVE is kept score-ordered (PRED-1); the other two are
        strict FIFO. A re-queued job asks for `head` and gets it in every class — see the
        livelock note on the class docstring."""
        dq = self._q[job.klass]
        if head:
            # A resumed job jumps the queue REGARDLESS of score — that is the livelock
            # protection, and it deliberately breaks score order at the front.
            job.resumed = True
            dq.appendleft(job)
        elif job.klass in SHEDDABLE:
            # Highest score first — but only over the SCORED region. The deque is
            # `resume-prefix (FIFO) ++ score-ordered tail`, so the insert scans past the
            # resumed jobs first: comparing against a low-scoring resumed head would stop
            # the scan at index 0 and let a new mid bet jump an already-queued higher one.
            # (With no policy installed every score is 0.0, so this falls through to the
            # same append the other classes get — score-ordering needs no separate guard.)
            i = 0
            while i < len(dq) and dq[i].resumed:
                i += 1
            while i < len(dq):
                if job.score > dq[i].score:
                    dq.insert(i, job)
                    return
                i += 1
            dq.append(job)
        else:
            dq.append(job)

    def _preempt_for_locked(self, klass: int) -> None:
        """Trip the running job's token if `klass` outranks it — unless the running job is
        protected by the starvation brake.

        THE BRAKE, and why it is not optional. A preempted cook loses ALL its partial progress
        (§4a: there is no resume). So unconditional preemption is not "the render pauses", it is
        "the render restarts", and if interactive requests arrive faster than the render takes,
        it **never finishes**. Measured: a 310 ms COMMITTED render against a 5 Hz slider — a user
        nudging one control — did not complete in 25 s across 11/11 trials: 104 attempts, 103
        requeues, 22.9 s of CPU burned for nothing, 73.8× the cost of the render itself. The rule
        is simply `T_render > mean interactive gap ⇒ never completes`, and CACHE-2 cannot save it
        (a cook that never finishes never populates a cache).

        Two bounds, both explicit constructor numbers a test can feed:

        * `min_quantum_ms` — a job that has just started is not preemptible. Bounds how much a
          burst of submits can shred, and stops a stampede of near-simultaneous slider frames
          from spending the whole budget in microseconds.
        * `max_preemptions` — after this many, a job runs to completion. This is what converts
          "never finishes" into "finishes, after at most one bounded wait".

        The honest cost, stated because it is a real trade: Tier A can now wait up to
        `min_quantum_ms`, and against a budget-exhausted job up to one full cook of it. That is
        the price of non-resumable cooks. Resumable cooks are the deferral that removes it, and
        the measurement above is the reopen gate its own design note asked for."""
        run = self._running
        if run is None or klass >= run.klass or run.preempt_requested or run.shed_requested:
            return
        if run.preemptions >= self.max_preemptions:
            # Budget spent: it finishes, or nothing does. This one IS a refusal rather than a
            # deferral — there is no later moment at which preempting it becomes the right call.
            self.stats.preempt_denied += 1
            return
        # The quantum is NOT checked here: the request is recorded now and the token honours it
        # once the quantum has elapsed. Checking here would DROP the request instead of
        # deferring it, and nothing would ever re-raise it.
        run.preempt_requested = True
        self.stats.preempted += 1

    # ── cancellation ─────────────────────────────────────────────────────────
    def cancel(self, job: Job) -> bool:
        """Cancel a job at the host's request. Terminal, for any class — this is the host
        saying "I no longer want this", which outranks COMMITTED's never-shed guarantee
        (that guarantee protects the render from the SCHEDULER, not from its owner).

        Returns True if the job was pending or running; False if it had already finished."""
        with self._wake:
            if job.state in _TERMINAL:
                return False
            job.shed_requested = True
            # P0-B: WAITING is cancellable exactly like PENDING, and leaving it out was not a
            # missing nicety — it was two defects. `job.wait()` and `drain()` hung forever on
            # a job the host had already given up on; and because `_wake_if_ready_locked` did
            # not consult `shed_requested`, the promise landing later flipped the cancelled
            # job back to PENDING and GRANTED ITS DEFERRED PREEMPTION — a running COMMITTED
            # render lost all its progress for a job that can never run, which then sat in the
            # deque until `close()`. `_finish_locked` owns the `waiting` counter, so this
            # needs no separate bookkeeping.
            if job.state in (PENDING, WAITING):
                self._remove_locked(job)
                self.stats.cancelled += 1
                self._finish_locked(job, CANCELLED,
                                    error=CookCancelled(f"job {job.id} cancelled before it ran"))
            self._wake.notify_all()
            return True

    def _remove_locked(self, job: Job) -> bool:
        try:
            self._q[job.klass].remove(job)
            return True
        except ValueError:
            return False

    def install_policy(self, policy) -> None:
        """Install PRED-1's admission policy (or None to remove it). Without one the queue
        admits everything and orders SPECULATIVE work FIFO — SCHED-4's own behaviour, and what
        the queue is tested against on its own."""
        with self._wake:
            self._policy = policy

    def _admit_locked(self, job: Job) -> bool:
        """True unless the installed policy refuses; True always when none is installed."""
        return True if self._policy is None else self._policy.admit(job)

    def _shed_locked(self) -> None:
        """Let the installed policy drop work under pressure. A no-op when none is installed."""
        if self._policy is not None:
            self._policy.shed(self)

    def shed_speculative(self, *, keep: int = 0) -> int:
        """Drop all but the `keep` highest-scoring PENDING speculative jobs. The pressure valve
        a memory governor or a host pulls; PRED-1's policy pulls it automatically on queue depth.

        The RUNNING job is never shed here, whatever its class. It has already paid for the
        work it has done, and Tier A preempts it anyway if something urgent arrives — killing
        it as well would spend the cost and keep none of the result."""
        with self._wake:
            n = self._shed_speculative_locked(keep)
            if n:
                self._wake.notify_all()
        return n

    def _shed_speculative_locked(self, keep: int) -> int:
        """The lock-held half, so PRED-1's `shed()` — which runs inside `submit()`, under the
        lock — can reach it without re-entering a non-reentrant Lock.

        Finds the victim by `min()`, NOT by popping the tail. An earlier version popped, on
        the reasoning that `_enqueue_locked` keeps the deque in descending score order so the
        tail is the minimum — and that reasoning was WRONG: head-requeue puts a preempted job
        at the front regardless of score, so once the tail grows past it the TAIL IS THE
        MAXIMUM, and the shed drops the single most valuable bet while keeping the worst.
        Reachable from the shipped `SpeculativePolicy` alone (its `shed()` runs on every
        submit), not just from the public valve. Two linear scans of a deque bounded by
        `max_pending` cost less than the linear insert `_enqueue_locked` already does per
        speculative submit, and `Job` is `eq=False` so `deque.remove` is an identity remove."""
        dropped = 0
        for klass in SHEDDABLE:
            dq = self._q[klass]
            while len(dq) > max(0, keep):
                job = min(dq, key=lambda j: j.score)
                dq.remove(job)
                job.shed_requested = True
                self.stats.shed += 1
                self.stats.cancelled += 1
                self._finish_locked(job, CANCELLED, error=CookCancelled(
                    f"speculative cook shed under pressure: {job.reason or 'no reason given'}"))
                dropped += 1
        return dropped

    # ── the worker ───────────────────────────────────────────────────────────
    def _ensure_worker(self) -> None:
        """Start the worker once. `_worker` is published only AFTER a successful `start()`:
        `Thread.start` can raise for real (thread exhaustion, interpreter shutdown), and
        since this method deliberately refuses to replace a non-None worker, publishing
        first left a never-started Thread in place and every later `submit()` hanging with
        no diagnostic — the exact failure `_run`'s own guard exists to prevent, reached
        through the one door it cannot cover."""
        with self._wake:
            if self._worker is not None or self._closed:
                return
            t = threading.Thread(target=self._run, name=self._name, daemon=True)
            self._worker = t
        try:
            t.start()
        except BaseException:
            with self._wake:
                if self._worker is t:
                    self._worker = None          # let the next submit try again
            raise

    def _next_locked(self) -> Job | None:
        """The highest-priority ELIGIBLE job, removed from its deque.

        IO-1 turned this from a `popleft` into a scan, because a WAITING job must be skipped
        and KEPT — popping it would drop it. The scan is what buys one home for every job:
        the shed loop, `cancel()`, `close()`'s drain and `snapshot()` all still see waiting
        jobs by looking exactly where they always looked. A side table would be cheaper by
        this scan and would hide a waiting speculative prefetch from the shed policy that
        exists to drop it.

        Cost: index 0 whenever nothing is waiting, which is every cook in the tree today —
        and `_enqueue_locked` already does a linear insert per speculative submit."""
        for k in CLASSES:                    # numeric order IS priority order
            for job in self._q[k]:
                if job.state == PENDING and not job.shed_requested:
                    self._q[k].remove(job)   # identity remove — Job is eq=False
                    return job
                # A non-PENDING, non-WAITING job in a deque is DEFENCE, not a live case:
                # every path that finishes a QUEUED job removes it first, under this same
                # lock. The scan keeps a future path that forgets that from handing the
                # worker a dead job.
        return None

    def _run(self) -> None:
        """The worker loop, wrapped so that NOTHING can kill this thread.

        It is the only worker, `_ensure_worker` deliberately will not replace a dead one (two
        workers would break the single-cook-thread contract ENG-9 rests on), and a dead worker
        means every waiter, every `drain()` and every `close()` hangs forever with no
        diagnostic. That was reachable with no monkeypatching at all — a host passing a `list`
        instead of a tuple as `profile_key` raised out of the PROF-1 feedback, which sits in the
        `else:` of `_run_one`'s try and so is covered by none of its handlers.

        Any unexpected escape is now logged, charged to the job that was running, and the loop
        continues."""
        while True:
            try:
                if self._loop_once():
                    return
            except BaseException:                        # noqa: BLE001 — a worker must not die
                logger.exception("[TEX] cook-queue worker recovered from an internal error")
                with self._wake:
                    run, self._running = self._running, None
                    if run is not None and run.state not in _TERMINAL:
                        self.stats.failed += 1
                        self._finish_locked(run, FAILED, error=RuntimeError(
                            "the cook queue's worker hit an internal error on this job"))
                    self._wake.notify_all()

    def _loop_once(self) -> bool:
        """Pop one job and run one attempt. True when the queue is closed and the worker exits."""
        with self._wake:
            while not self._closed and (job := self._next_locked()) is None:
                self._wake.wait()
            if self._closed:
                self._running = None
                self._wake.notify_all()
                return True
            job.state = RUNNING
            job.attempts += 1
            job.started_at = time.perf_counter()
            # A fresh attempt starts un-tripped, and leaves the resume prefix.
            job.preempt_requested = job.resumed = False
            self._running = job
            token = _JobToken(job, self.min_quantum_ms)
        self._run_one(job, token)
        # DO NOT inline this method back into `_run`. `job` and `token` reach the Job's `fn` (a
        # closure over the input frame) and its `value` (a CookResult holding the outputs); the
        # worker used to hold both across `_wake.wait()`, pinning one whole cook's input AND
        # output tensors for the entire idle gap between drags — VRAM the allocator cannot
        # reuse. Returning here drops them with the frame, and the next call parks in a frame
        # that never saw them.
        return False

    def _run_one(self, job: Job, token: _JobToken) -> None:
        """Run one attempt OUTSIDE the lock — a cook is milliseconds to seconds, and holding
        the lock across it would make `submit()` (and therefore preemption) block on the very
        job it is trying to preempt."""
        try:
            value = job.fn(token)
        except _Preempted:
            # OUR token raised it: transient. `shed_requested` still wins — a `close()` or a
            # host `cancel()` can land between the raise and here, and a teardown outranks a
            # re-queue.
            with self._wake:
                self._running = None
                if job.shed_requested:
                    self.stats.cancelled += 1
                    self._finish_locked(job, CANCELLED,
                                        error=CookCancelled("cook shed while yielding"))
                else:
                    job.state = PENDING          # the waiter is NOT woken
                    job.preempt_requested = False
                    job.preemptions += 1
                    self.stats.requeued += 1
                    self._enqueue_locked(job, head=True)
                self._wake.notify_all()
        except CookCancelled as e:
            # A cancellation the queue did NOT cause — a shed, a host's supersede latch, a
            # global Stop. Terminal by definition: the queue does not get to overrule the host,
            # and re-running a job whose token is still tripped is the livelock this ordering
            # exists to make impossible.
            with self._wake:
                self._running = None
                self.stats.cancelled += 1
                self._finish_locked(job, CANCELLED, error=e)
                self._wake.notify_all()
        except BaseException as e:                     # noqa: BLE001 — the cook's error is the host's
            # DATA-7: a HOST-I/O failure (E7xxx) is class-dependent, and the provider cannot
            # know its job's class — so the decision lives here, where the class is known. An
            # INTERACTIVE or COMMITTED job reports it: the user asked for that frame and is
            # owed the reason it did not arrive. A SPECULATIVE one dies into the refusals
            # ledger instead, because a prefetch that alarms about a source the user never
            # asked for is worse than a prefetch that quietly did not happen.
            if job.klass == SPECULATIVE and str(getattr(e, "_code", "")).startswith("E7"):
                with self._wake:
                    self._running = None
                    self._note_speculative_io_failure(job, e)
                    self.stats.cancelled += 1
                    self._finish_locked(job, CANCELLED, error=CookCancelled(
                        f"speculative host-I/O failure ({getattr(e, '_code', 'E7')}): {e}"))
                    self._wake.notify_all()
                return
            with self._wake:
                self._running = None
                self.stats.failed += 1
                self._finish_locked(job, FAILED, error=e)
                self._wake.notify_all()
        else:
            # PRED-1 ↔ PROF-1: the queue already brackets every job, so a submitter who named
            # a profile key gets that measurement fed back for free — no engine involvement,
            # no sampling gate, nothing on the default cook path. This closes the loop: the
            # next speculative submit of the same program is scored on a measured cost.
            # Outside the lock; `record` is the profiler's own bounded LRU.
            #
            # WHAT THIS NUMBER IS, stated because it is NOT the engine's: it is the wall time
            # of the whole submitted `fn`, including whatever fast paths the host has of its
            # own (a frame served from the host's cache genuinely costs ~0). That is the right
            # quantity for admission — PRED-1 is pricing "what does submitting this job cost",
            # not "what does computing this program cost". The engine's `measure` feeds the
            # second question, and CACHE-7's input is the per-STAGE table, which only the
            # engine writes.
            if job.profile_key is not None and job.feeds_profile:
                try:
                    _profile.record(job.profile_key,
                                    (time.perf_counter() - job.started_at) * 1000.0, job.px)
                except BaseException:            # noqa: BLE001
                    pass                         # a profiler feed never costs a RESULT
            # §4b: the cook RETURNED. Complete it with that value whatever flags were raised
            # while it ran — a finished frame is never thrown away for a preempt that lost
            # the race. This `else` (not a `finally`, not a post-cook flag check) is the rule.
            with self._wake:
                self._running = None
                self.stats.completed += 1
                self._finish_locked(job, DONE, value=value)
                self._wake.notify_all()

    def _note_speculative_io_failure(self, job: Job, exc: BaseException) -> None:
        """Record a shed-by-I/O in the policy's refusals ledger. Lock held.

        Reaches `_refuse` rather than re-implementing the counter, so a host reading
        `policy.refusals` sees prefetch failures grouped by the same reason string it groups
        admission refusals by — one ledger, one question ("why did my prefetches stop?"),
        one answer. Silent when no policy is installed: with nobody scoring speculation
        there is nobody to report to."""
        policy = self._policy
        refuse = getattr(policy, "_refuse", None)
        if refuse is None:
            return
        try:
            refuse(job, f"host I/O failed: {getattr(exc, '_code', '')} {exc}".strip())
        except BaseException:                # noqa: BLE001 — a host policy bug is not ours
            logger.exception("[TEX] cook-queue refusals ledger raised; ignoring")

    def _finish_locked(self, job: Job, state: str, *, value=None, error=None) -> None:
        # IO-1: a WAITING job can be finished by four different paths (a failed promise, a
        # shed, a host cancel, close()'s drain). Decrementing here — the one door all four go
        # through — is why `stats.waiting` cannot drift, and a drifted counter is exactly the
        # symptom it exists to report.
        if job.state == WAITING:
            self.stats.waiting -= 1
        job.state = state
        job.value = value
        job.error = error
        job._done.set()

    # ── lifecycle / introspection ────────────────────────────────────────────
    @property
    def running(self) -> Job | None:
        return self._running

    def snapshot(self) -> dict:
        """A stats blob for a host HUD or a test: counters plus per-class queue depth."""
        with self._wake:
            run = self._running
            return {"stats": self.stats.as_dict(),
                    "pending": {CLASS_NAMES[k]: sum(1 for j in self._q[k]
                                                    if j.state == PENDING)
                                for k in CLASSES},
                    "waiting": {CLASS_NAMES[k]: sum(1 for j in self._q[k]
                                                    if j.state == WAITING)
                                for k in CLASSES},
                    "running": None if run is None else
                               {"id": run.id, "class": run.class_name, "attempts": run.attempts}}

    def drain(self, timeout: float | None = None) -> bool:
        """Block until nothing is queued or running. True if it drained, False on timeout.

        ON A CLOSED QUEUE this returns as soon as the backlog is gone, which `close()` empties
        synchronously — so it can return True with the last job still RUNNING. That is
        deliberate: after `close()` the backlog is what a caller can still wait for, and the
        running cook is the one thing the queue explicitly cannot stop (see `close`). Use
        `close(timeout=…)` to wait for that one; it joins the worker."""
        deadline = None if timeout is None else time.perf_counter() + timeout
        with self._wake:
            while self._running is not None or any(self._q[k] for k in CLASSES):
                if self._closed:
                    return True
                remain = None if deadline is None else deadline - time.perf_counter()
                if remain is not None and remain <= 0:
                    return False
                self._wake.wait(remain)      # None blocks until notified; every
                #                                  transition notifies, so there is nothing to
                #                                  poll for and a fallback tick is pure noise
            return True

    def close(self, *, timeout: float | None = 2.0) -> None:
        """Stop the worker and cancel everything still queued. A RUNNING job is asked to yield
        (its token trips) but is never killed — there is no safe way to abort a torch kernel
        from another thread, and pretending otherwise would corrupt the CUDA stream.

        IDEMPOTENT AND FULLY BLOCKING for every caller, not just the first. An earlier version
        returned immediately when `_closed` was already set, so a second `close()` — an explicit
        one plus `__exit__`, or two threads racing shutdown — returned claiming the queue was
        down while the worker was still mid-cook (measured: 3 of 4 concurrent callers)."""
        with self._wake:
            if self._closed:
                t = self._worker                 # already closing: still wait for the worker
                if t is not None and t.is_alive() and t is not threading.current_thread():
                    self._wake.release()
                    try:
                        t.join(timeout)
                    finally:
                        self._wake.acquire()
                return
            self._closed = True
            run = self._running
            if run is not None:
                run.shed_requested = True
            for k in CLASSES:
                while self._q[k]:
                    job = self._q[k].popleft()
                    if job.state not in _TERMINAL:
                        self.stats.cancelled += 1
                        self._finish_locked(job, CANCELLED,
                                            error=CookCancelled("CookQueue closed"))
            self._wake.notify_all()
            t = self._worker
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout)

    def __enter__(self) -> "CookQueue":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ── PRED-1: the speculative-cook protocol ────────────────────────────────────
#
# A host that can predict what the user is about to need should be able to say so, and the
# engine should be able to disagree. PRED-1 is that conversation: the host supplies a
# CONFIDENCE and a REASON, TEX supplies the cost (PROF-1) and the arithmetic.
#
# THE SPLIT, stated once (doc 39 §7 puts prediction policy in host territory): **the host owns
# the psychology, TEX owns the economics.** Which UI signal means 0.8 and which means 0.2 is a
# question about users, and only a real session can answer it — so `confidence` is an input,
# never something TEX infers from `reason`. What TEX owns is what to do with a number once it
# has one, and that is not a matter of taste.
#
# THE ARITHMETIC. Speculating on a job is a bet: pay `cost` of worker time now for a
# `confidence` chance of saving `cost` of latency later. Expected saving is therefore
# `confidence × cost`, and that single quantity both ADMITS (is this bet worth taking at all?)
# and ORDERS (which bet first?). It has the two properties the policy needs:
#
#   * a job nobody will probably want scores low however expensive it is — no speculating on
#     a 400 ms render at confidence 0.02;
#   * a job everyone will want scores low if it is CHEAP — a 0.3 ms grade at confidence 1.0 is
#     not worth a queue slot, because cooking it on demand costs the user nothing visible.
#
# The second is the counter-intuitive half and the reason the rule is `confidence × cost`
# rather than a confidence threshold: the point of speculation is to hide *latency the user
# would otherwise feel*, and there is none to hide on work that is already instant.

# The reasons doc 39 names, plus the two the demo host uses. OPEN, not closed — a host may
# pass any string and TEX never reads it except to report it. Named so hosts converge on
# spellings a HUD and a log can group by, rather than each inventing its own.
PANEL_OPEN = "panel-open"
PLAY_HOVER = "play-hover"
NEIGHBOR_FRAME = "neighbor-frame"
IDLE_CHECKPOINT = "idle-checkpoint"
PREFETCH = "prefetch"


class SpeculativePolicy:
    """PRED-1's admission + shed policy for the SPECULATIVE class. Install it on a queue:

        q.install_policy(SpeculativePolicy())
        q.submit(fn, klass=SPECULATIVE, reason="play-hover", confidence=0.7,
                 profile_key=_profile.make_key(fp, "cuda", "fp32"), px=1024*1024)

    INTERACTIVE and COMMITTED are never scored, never refused and never shed — they were
    asked for. Every constant is a constructor argument so a test feeds explicit numbers
    instead of reverse-engineering them, which is autotier's discipline and the reason its
    verdicts are testable at all."""

    def __init__(self, *, min_value_ms: float = 2.0, max_pending: int = 8,
                 min_confidence: float = 0.15, max_cost_ms: float = 250.0,
                 unknown_cost_ms: float = 8.0, unknown_min_confidence: float = 0.5,
                 predict=None):
        #: Refuse a speculative bet worth less than this many ms of expected saving.
        self.min_value_ms = float(min_value_ms)
        #: …and refuse one below this confidence WHATEVER it would save. Without it the product
        #: rule alone admits the CHANGELOG's own counter-example: confidence 0.02 × 400 ms scores
        #: 8.0, clears a 2 ms floor, and TEX cheerfully spends 400 ms of worker time on a 2%
        #: chance. `confidence × cost` ranks bets correctly and bounds neither factor, so each
        #: factor needs its own bound.
        self.min_confidence = float(min_confidence)
        #: The other unbounded factor: refuse a single speculative cook longer than this,
        #: however likely. A 30-second background render is `COMMITTED` work a host should submit
        #: as such, not something to guess at.
        self.max_cost_ms = float(max_cost_ms)
        #: Keep at most this many speculative jobs queued; the lowest-scoring are shed.
        self.max_pending = int(max_pending)
        #: What an UNMEASURED program is assumed to cost. A cold session has no PROF-1 samples
        #: and refusing everything would mean it never cooks one, never measures one, and never
        #: learns — so an unknown scores at a neutral middle rather than at zero.
        self.unknown_cost_ms = float(unknown_cost_ms)
        #: …but an unmeasured guess is only accepted from a host that is fairly sure. This is
        #: the brake on the optimism above.
        self.unknown_min_confidence = float(unknown_min_confidence)
        #: Cost oracle seam: `predict(profile_key, px) -> ms | None`. Defaults to PROF-1.
        self._predict = predict or _profile.predict
        #: reason -> (count, latest explanation), for a host HUD / a test.
        self.refusals: dict = {}

    # ── the cost half ──
    def cost_of(self, job: Job) -> tuple:
        """(predicted_ms, measured) for a job. `measured` is False when the number is the
        `unknown_cost_ms` placeholder — the caller needs to know, because an unmeasured job
        faces the extra confidence brake."""
        if job.cost_ms is not None:
            return float(job.cost_ms), True
        if job.profile_key is not None:
            ms = self._predict(job.profile_key, job.px)
            if ms is not None and ms > 0:
                return float(ms), True
        return self.unknown_cost_ms, False

    @staticmethod
    def _value(confidence: float, cost_ms: float) -> float:
        """The rule, spelled ONCE: expected saving = clamped confidence × predicted cost."""
        return max(0.0, min(1.0, confidence)) * cost_ms

    def score_of(self, job: Job) -> float:
        """Expected saving in ms for a job. See the arithmetic note above."""
        return self._value(job.confidence, self.cost_of(job)[0])

    # ── the CookQueue policy surface ──
    def admit(self, job: Job) -> bool:
        if job.klass != SPECULATIVE:
            job.score = float("inf")          # sorts first if it ever reaches a scored deque
            return True
        cost, measured = self.cost_of(job)
        job.score = self._value(job.confidence, cost)
        # Each factor is bounded on its own, THEN the product. The product alone is a ranking
        # rule, not an admission rule — it says nothing about a 2%-likely 400 ms render, which
        # scores well above any sane floor.
        if job.confidence < self.min_confidence:
            return self._refuse(job, f"confidence {job.confidence:.3f} < {self.min_confidence}")
        if cost > self.max_cost_ms:
            return self._refuse(job, f"a single {cost:.0f} ms cook is too long to guess at "
                                     f"(> {self.max_cost_ms} ms) — submit it as COMMITTED")
        if not measured and job.confidence < self.unknown_min_confidence:
            return self._refuse(job, "unmeasured program below the confidence brake")
        if job.score < self.min_value_ms:
            return self._refuse(job, f"expected saving {job.score:.2f} ms < {self.min_value_ms} ms")
        return True

    def _refuse(self, job: Job, why: str) -> bool:
        """Count the refusal by REASON (what a host groups a HUD by) and keep the latest
        explanation for it. Building the message and discarding it would leave a host asking
        "why did my prefetches stop?" with a counter and no answer."""
        key = job.reason or "unspecified"
        n, _prev = self.refusals.get(key, (0, ""))
        self.refusals[key] = (n + 1, why)
        return False

    def shed(self, queue: "CookQueue") -> None:
        """Called under the queue's lock after each admission. Depth IS the pressure signal
        here; a memory governor with a better one calls `queue.shed_speculative(keep=...)`."""
        queue._shed_speculative_locked(self.max_pending)
