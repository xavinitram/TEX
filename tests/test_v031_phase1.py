"""v0.31 Phase 1 — SCHED-4: the two-tier preemptive cook queue.

What has to be pinned, and why each one is a bug the design note names:

  * PRIORITY + PREEMPTION — an INTERACTIVE submit trips a running SPECULATIVE cook at its
    next SCHED-3 yield point and runs first. This is the item.
  * THE §4b RULE — a cook that RETURNS while a preempt flag is raised completes with its
    value. The obvious implementation (worker checks the flag after the cook) throws away
    finished frames under exactly the load where frames are most expensive.
  * PREEMPT != SHED — a preempted job is re-queued (transient, waiter not woken); a shed or
    cancelled job is terminal. Collapsing the two into one `cancelled` flag is how a
    scheduler silently loses work.
  * COMMITTED IS NEVER SHED — the user started that render by name.
  * HEAD-REQUEUE / NO LIVELOCK — a preempted job goes to the head of its class, so sustained
    Tier-A pressure does not cycle its peers forever while it never finishes.
  * INVARIANT #7 — the ComfyUI cook path neither imports nor reaches the queue, and no thread
    exists until a host submits.

Shapes (roadmap §10.4): CANARY for the class table and the invariant-#7 contract, NEVER-SEVER
ROWS for the preempt/shed/cancel outcome matrix, and one real `tex_engine.cook` integration
row so none of the above can pass vacuously against a synthetic callable.
"""
import threading

from helpers import *

from TEX_Wrangle import tex_cookqueue as Q
from TEX_Wrangle.tex_runtime.host import CookCancelled

_WAIT = 10.0          # generous: these are thread rendezvous, not timing assertions


def _blocker(started: threading.Event, release: threading.Event):
    """A job body that parks at a SCHED-3-shaped yield loop: it announces that it is running,
    then polls the token until the test releases it. Polling IS what a real cook does at its
    yield points, so a preempt lands here exactly as it would land between two statements."""
    def fn(cancel):
        started.set()
        while not release.wait(0.002):
            cancel.check()
        cancel.check()
        return "blocked-work"
    return fn


def test_v031_sched4_priority_and_preemption(r: SubTestResult):
    print("\n--- v0.31 SCHED-4: priority admission + preemption ---")
    started, release = threading.Event(), threading.Event()
    with Q.CookQueue() as q:
        spec = q.submit(_blocker(started, release), klass=Q.SPECULATIVE, reason=Q.NEIGHBOR_FRAME)
        if not started.wait(_WAIT):
            r.fail("SCHED-4 preempt", "speculative job never started")
            return

        order = []
        inter = q.submit(lambda cancel: order.append("interactive") or "frame",
                         klass=Q.INTERACTIVE)
        # The interactive job must complete WITHOUT the speculative job being released — that
        # is the whole claim. If preemption did not work this blocks until the timeout.
        try:
            got = inter.result(timeout=_WAIT)
        except TimeoutError:
            r.fail("SCHED-4 preempt", "interactive job never ran while speculative held the worker")
            release.set()
            return
        r.ok("interactive preempts a running speculative cook") if got == "frame" else \
            r.fail("SCHED-4 preempt", f"interactive returned {got!r}")

        s = q.snapshot()["stats"]
        r.ok(f"preempt + requeue counted (preempted={s['preempted']}, requeued={s['requeued']})") \
            if s["preempted"] == 1 and s["requeued"] == 1 else \
            r.fail("SCHED-4 counters", f"preempted={s['preempted']} requeued={s['requeued']}")

        # The preempted job is back in play, not cancelled, and its waiter was never woken.
        # (PENDING *or* RUNNING: the worker may already have picked the re-queued job up by
        # now — the claim is that it is not terminal, not that we caught it mid-air.)
        r.ok(f"preempted job is re-queued, not lost (state={spec.state})") \
            if spec.state in (Q.PENDING, Q.RUNNING) and not spec._done.is_set() else \
            r.fail("SCHED-4 requeue", f"state={spec.state} done={spec._done.is_set()}")

        release.set()
        try:
            v = spec.result(timeout=_WAIT)
        except (TimeoutError, CookCancelled) as e:
            r.fail("SCHED-4 resume", f"re-queued job did not finish: {e!r}")
            return
        r.ok(f"re-queued job resumes and completes (attempts={spec.attempts})") \
            if v == "blocked-work" and spec.attempts == 2 else \
            r.fail("SCHED-4 resume", f"value={v!r} attempts={spec.attempts}")


def test_v031_sched4_finished_work_is_never_discarded(r: SubTestResult):
    """The §4b rule, probed directly: preemption is cooperative, so a cook can RETURN after the
    flag was set but before the next yield. That result is real and paid for."""
    print("\n--- v0.31 SCHED-4: a completed cook survives a preempt that lost the race ---")
    raised = threading.Event()

    def racy(cancel):
        # Park until the test has definitely raised the preempt flag, then return WITHOUT
        # polling again — the race the rule is about.
        raised.wait(_WAIT)
        return "expensive-frame"

    with Q.CookQueue() as q:
        job = q.submit(racy, klass=Q.SPECULATIVE)
        for _ in range(int(_WAIT / 0.002)):          # wait for the worker to pick it up
            if q.running is not None and q.running.id == job.id:
                break
            time.sleep(0.002)
        if q.running is None or q.running.id != job.id:
            r.fail("SCHED-4 race", "job never reached RUNNING")
            return
        q.submit(lambda cancel: "urgent", klass=Q.INTERACTIVE)   # sets preempt_requested
        r.ok("preempt flag raised on the running job") if job.preempt_requested else \
            r.fail("SCHED-4 race", "preempt flag was not set")
        raised.set()
        try:
            v = job.result(timeout=_WAIT)
        except (TimeoutError, CookCancelled) as e:
            r.fail("SCHED-4 race", f"a completed cook was discarded: {e!r}")
            return
        r.ok("a cook that returned under a raised preempt flag completes DONE") \
            if v == "expensive-frame" and job.state == Q.DONE and job.attempts == 1 else \
            r.fail("SCHED-4 race", f"value={v!r} state={job.state} attempts={job.attempts}")


def test_v031_sched4_outcome_matrix(r: SubTestResult):
    """NEVER-SEVER ROWS: preempt / shed / cancel / close each have ONE correct outcome."""
    print("\n--- v0.31 SCHED-4: preempt vs shed vs cancel are distinct outcomes ---")

    # (1) cancel a PENDING job -> terminal CANCELLED, never runs.
    started, release = threading.Event(), threading.Event()
    with Q.CookQueue() as q:
        hog = q.submit(_blocker(started, release), klass=Q.INTERACTIVE)
        started.wait(_WAIT)
        ran = []
        waiting = q.submit(lambda cancel: ran.append(1), klass=Q.INTERACTIVE)
        r.ok("cancel() on a pending job returns True") if q.cancel(waiting) else \
            r.fail("SCHED-4 cancel", "cancel(pending) returned False")
        release.set()
        hog.result(timeout=_WAIT)
        q.drain(timeout=_WAIT)
        r.ok("a cancelled pending job is terminal CANCELLED and never runs") \
            if waiting.state == Q.CANCELLED and not ran else \
            r.fail("SCHED-4 cancel", f"state={waiting.state} ran={ran}")
        try:
            waiting.result(timeout=0.1)
            r.fail("SCHED-4 cancel", "result() on a cancelled job did not raise")
        except CookCancelled:
            r.ok("result() on a cancelled job raises CookCancelled")

    # (2) cancel a RUNNING job -> terminal, NOT re-queued (the preempt path must not catch it).
    started, release = threading.Event(), threading.Event()
    with Q.CookQueue() as q:
        job = q.submit(_blocker(started, release), klass=Q.COMMITTED)
        started.wait(_WAIT)
        q.cancel(job)
        try:
            job.result(timeout=_WAIT)
            r.fail("SCHED-4 cancel-running", "a cancelled running job completed")
        except CookCancelled:
            r.ok("cancel() on a running job is terminal, not a re-queue") \
                if job.state == Q.CANCELLED and q.snapshot()["stats"]["requeued"] == 0 else \
                r.fail("SCHED-4 cancel-running",
                       f"state={job.state} requeued={q.snapshot()['stats']['requeued']}")
        except TimeoutError:
            r.fail("SCHED-4 cancel-running", "a cancelled running job never finished")
        release.set()

    # (3) close() cancels the backlog and asks the runner to yield.
    started, release = threading.Event(), threading.Event()
    q = Q.CookQueue()
    running = q.submit(_blocker(started, release), klass=Q.COMMITTED)
    started.wait(_WAIT)
    queued = q.submit(lambda cancel: "never", klass=Q.SPECULATIVE)
    q.close(timeout=_WAIT)
    release.set()
    r.ok("close() cancels the backlog") if queued.state == Q.CANCELLED else \
        r.fail("SCHED-4 close", f"queued job state={queued.state}")
    running.wait(_WAIT)
    r.ok("close() asks the running job to yield (terminal, not re-queued)") \
        if running.state == Q.CANCELLED else \
        r.fail("SCHED-4 close", f"running job state={running.state}")


def test_v031_sched4_a_foreign_cancel_is_terminal(r: SubTestResult):
    """NEVER-SEVER ROW, and the one that would have shipped a hang.

    A cook can raise `CookCancelled` for a reason the queue knows nothing about — a host's
    supersede latch, a global Stop, a chained token. Reading "not shed, therefore preempted"
    requeues that job at the head of its class, where its still-tripped token raises again on
    the first yield, forever: measured at ~450k requeues per second, starving every other job
    in the queue. Only a preemption the queue ITSELF asked for is transient."""
    print("\n--- v0.31 SCHED-4: a cancel the queue did not cause is terminal ---")
    from TEX_Wrangle.tex_runtime.host import CookCancelled as _CC

    ran = []
    with Q.CookQueue() as q:
        # A job whose own token is permanently tripped, the way a superseded frame's is.
        doomed = q.submit(lambda cancel: (_ for _ in ()).throw(_CC("superseded by the host")),
                          klass=Q.SPECULATIVE, reason=Q.PLAY_HOVER)
        q.submit(lambda cancel: ran.append("peer"), klass=Q.SPECULATIVE)
        try:
            doomed.result(timeout=_WAIT)
            r.fail("SCHED-4 foreign cancel", "a host-cancelled job completed")
        except _CC:
            r.ok("a host-raised CookCancelled finishes the job CANCELLED, not re-queued") \
                if doomed.state == Q.CANCELLED and doomed.attempts == 1 else \
                r.fail("SCHED-4 foreign cancel",
                       f"state={doomed.state} attempts={doomed.attempts}")
        except TimeoutError:
            r.fail("SCHED-4 foreign cancel", "livelock: the job never reached a terminal state")

        q.drain(timeout=_WAIT)
        st = q.snapshot()["stats"]
        r.ok(f"it does not starve the queue (peer ran, requeued={st['requeued']})") \
            if ran == ["peer"] and st["requeued"] == 0 else \
            r.fail("SCHED-4 foreign cancel", f"ran={ran} requeued={st['requeued']}")

    # THE HALF THAT WAS STILL BROKEN. Testing `preempt_requested` instead of the token's own
    # stamp misclassifies a foreign cancel that lands WHILE a preempt is outstanding — the flag
    # is set, our token never raised, and the job is requeued forever against a token that can
    # never stop raising. So: arrange exactly that overlap.
    running, release = threading.Event(), threading.Event()

    def foreign_after_preempt(cancel):
        running.set()
        release.wait(_WAIT)            # the test raises a preempt while we sit here…
        raise _CC("the host superseded this, and our token never raised")

    with Q.CookQueue(min_quantum_ms=0.0) as q:
        job = q.submit(foreign_after_preempt, klass=Q.SPECULATIVE)
        if not running.wait(_WAIT):
            r.fail("SCHED-4 foreign cancel", "the job never started")
            return
        q.submit(lambda cancel: "urgent", klass=Q.INTERACTIVE)     # sets preempt_requested
        r.ok("a preempt is outstanding on the running job") if job.preempt_requested else \
            r.fail("SCHED-4 foreign cancel", "no preempt was raised")
        release.set()
        try:
            job.result(timeout=_WAIT)
            r.fail("SCHED-4 foreign cancel", "the foreign-cancelled job completed")
        except _CC:
            r.ok("a foreign cancel landing UNDER an outstanding preempt is still terminal") \
                if job.state == Q.CANCELLED and job.attempts == 1 else \
                r.fail("SCHED-4 foreign cancel",
                       f"requeued a cancel our token never raised: state={job.state} "
                       f"attempts={job.attempts}")
        except TimeoutError:
            r.fail("SCHED-4 foreign cancel",
                   "livelock: requeued forever against a permanently-tripped token")


def test_v031_sched4_committed_render_completes_under_load(r: SubTestResult):
    """The starvation brake. A preempted cook loses ALL its progress (there is no resume), so
    unconditional preemption does not pause a render — it restarts it. If interactive requests
    arrive faster than the render takes, it NEVER FINISHES.

    Both halves are asserted, because the brake is only meaningful against the failure: a
    wide-open queue must starve the render, and the shipped default must complete it. Measured
    against a real engine cook at 1024²: brake off, a 460 ms render made 156 attempts and 0
    completions in 25 s; brake on, it finished in 1.2 s after 4 attempts."""
    print("\n--- v0.31 SCHED-4: a COMMITTED render finishes under sustained Tier-A load ---")
    gate = threading.Event()

    def render(cancel):
        """~600 ms of work with frequent yield points — the shape of a real multi-stage cook."""
        for _ in range(300):
            cancel.check()
            time.sleep(0.002)
        return "render"

    def hammer(q, stop, gap):
        while not stop.wait(gap):
            q.submit(lambda c: None, klass=Q.INTERACTIVE)

    for label, kw, want in (("wide open (pre-fix)", dict(min_quantum_ms=0.0,
                                                        max_preemptions=10 ** 9), False),
                            ("shipped default", {}, True)):
        with Q.CookQueue(**kw) as q:
            job = q.submit(render, klass=Q.COMMITTED, reason="user render")
            stop = threading.Event()
            t = threading.Thread(target=hammer, args=(q, stop, 0.05), daemon=True)
            t.start()
            done = job.wait(6.0)
            stop.set()
            t.join(2.0)
            st = q.snapshot()["stats"]
            if done == want:
                r.ok(f"{label}: completed={done} (attempts={job.attempts}, "
                     f"requeued={st['requeued']}, denied={st['preempt_denied']})")
            else:
                r.fail("SCHED-4 starvation",
                       f"{label}: completed={done}, wanted {want} "
                       f"(attempts={job.attempts} requeued={st['requeued']})")

    r.ok("the brake is configurable (min_quantum_ms + max_preemptions)") \
        if Q.CookQueue().min_quantum_ms > 0 and Q.CookQueue().max_preemptions >= 1 else \
        r.fail("SCHED-4 starvation", "the shipped defaults do not brake")


def test_v031_sched4_worker_survives_a_bad_submit(r: SubTestResult):
    """The worker is the ONLY one and `_ensure_worker` will not replace it, so if it dies every
    waiter, `drain()` and `close()` hang forever with no diagnostic.

    The trigger needs no monkeypatching: a host passing a `list` instead of a tuple as
    `profile_key` made `_profile.record` raise from the `else:` of the run try — covered by none
    of its handlers — and killed the thread."""
    print("\n--- v0.31 SCHED-4: nothing can kill the worker ---")
    with Q.CookQueue() as q:
        bad = q.submit(lambda cancel: "fine", klass=Q.INTERACTIVE,
                       profile_key=["not", "a", "tuple"], px=4096)
        try:
            got = bad.result(timeout=_WAIT)
        except (TimeoutError, CookCancelled) as e:
            r.fail("SCHED-4 worker death", f"an unhashable profile_key lost the result: {e!r}")
            return
        r.ok("an unhashable profile_key does not cost the RESULT") if got == "fine" else \
            r.fail("SCHED-4 worker death", f"got {got!r}")
        try:
            nxt = q.submit(lambda cancel: "second", klass=Q.INTERACTIVE).result(timeout=_WAIT)
        except TimeoutError:
            r.fail("SCHED-4 worker death", "the worker died; every later job hangs")
            return
        r.ok("the worker is still serving afterwards") if nxt == "second" else \
            r.fail("SCHED-4 worker death", f"got {nxt!r}")
        r.ok("the worker thread is alive") if q._worker.is_alive() else \
            r.fail("SCHED-4 worker death", "thread is dead")

    # And an outright internal error inside the loop is charged to the job, not to the thread.
    with Q.CookQueue() as q:
        boom = q.submit(lambda cancel: (_ for _ in ()).throw(ValueError("cook blew up")),
                        klass=Q.INTERACTIVE)
        try:
            boom.result(timeout=_WAIT)
            r.fail("SCHED-4 worker death", "a raising cook did not surface its error")
        except ValueError:
            r.ok("a raising cook surfaces as FAILED, not as a dead worker")
        except TimeoutError:
            r.fail("SCHED-4 worker death", "a raising cook hung the queue")
        r.ok("…and the queue keeps serving") \
            if q.submit(lambda c: "ok", klass=Q.INTERACTIVE).result(timeout=_WAIT) == "ok" else \
            r.fail("SCHED-4 worker death", "the queue stopped after a raising cook")


def test_v031_sched4_class_contract(r: SubTestResult):
    """CANARY over the admission-class table. These three facts are the item's API, and doc 39
    is explicit that two classes cannot express the workload."""
    print("\n--- v0.31 SCHED-4: the three admission classes ---")
    r.ok("classes are priority-ordered ints (lower outranks)") \
        if Q.INTERACTIVE < Q.COMMITTED < Q.SPECULATIVE and Q.CLASSES == (0, 1, 2) else \
        r.fail("SCHED-4 classes", f"{Q.CLASSES}")
    r.ok("SPECULATIVE is the only sheddable class (COMMITTED is never shed)") \
        if Q.SHEDDABLE == frozenset({Q.SPECULATIVE}) else \
        r.fail("SCHED-4 sheddable", f"{Q.SHEDDABLE}")

    # A COMMITTED job pauses under INTERACTIVE but is never dropped.
    started, release = threading.Event(), threading.Event()
    with Q.CookQueue() as q:
        com = q.submit(_blocker(started, release), klass=Q.COMMITTED, reason="user render")
        started.wait(_WAIT)
        q.submit(lambda cancel: "frame", klass=Q.INTERACTIVE).result(timeout=_WAIT)
        # Paused, not shed: it yielded (preempted==1) and it is still not terminal. Reading
        # `state == PENDING` here would race the worker picking the re-queued job back up.
        paused = q.snapshot()["stats"]["preempted"] == 1 and not com._done.is_set()
        release.set()
        try:
            v = com.result(timeout=_WAIT)
        except (TimeoutError, CookCancelled) as e:
            r.fail("SCHED-4 committed", f"a committed render was lost: {e!r}")
            return
        r.ok("COMMITTED pauses under INTERACTIVE and then completes") \
            if paused and v == "blocked-work" else \
            r.fail("SCHED-4 committed", f"paused={paused} value={v!r}")

    # A SPECULATIVE submit never preempts a running COMMITTED job.
    started, release = threading.Event(), threading.Event()
    with Q.CookQueue() as q:
        com = q.submit(_blocker(started, release), klass=Q.COMMITTED)
        started.wait(_WAIT)
        q.submit(lambda cancel: "spec", klass=Q.SPECULATIVE)
        time.sleep(0.05)
        no_preempt = not com.preempt_requested and q.snapshot()["stats"]["preempted"] == 0
        release.set()
        q.drain(timeout=_WAIT)
        r.ok("a lower class never preempts a running higher one") if no_preempt else \
            r.fail("SCHED-4 no-inversion", "a speculative submit preempted a committed cook")


def test_v031_sched4_fifo_and_head_requeue(r: SubTestResult):
    """Ordering: FIFO within a class; a preempted job returns to the HEAD of its class, so
    sustained Tier-A pressure cannot livelock it behind its own peers."""
    print("\n--- v0.31 SCHED-4: FIFO within a class, head-requeue on preempt ---")
    gate, order = threading.Event(), []
    with Q.CookQueue() as q:
        hog = q.submit(lambda cancel: gate.wait(_WAIT), klass=Q.INTERACTIVE)
        time.sleep(0.02)
        jobs = [q.submit((lambda n: lambda cancel: order.append(n))(i), klass=Q.COMMITTED)
                for i in range(4)]
        gate.set()
        q.drain(timeout=_WAIT)
        r.ok(f"FIFO within a class: {order}") if order == [0, 1, 2, 3] else \
            r.fail("SCHED-4 fifo", f"{order}")

    # Head-requeue: preempt A once, then queue three peers behind it. A must run BEFORE them.
    started, release, seen = threading.Event(), threading.Event(), []

    def victim(cancel):
        started.set()
        while not release.wait(0.002):
            cancel.check()
        seen.append("victim")
        return "v"

    with Q.CookQueue() as q:
        v = q.submit(victim, klass=Q.SPECULATIVE)
        started.wait(_WAIT)
        q.submit(lambda cancel: "urgent", klass=Q.INTERACTIVE).result(timeout=_WAIT)
        peers = [q.submit((lambda n: lambda cancel: seen.append(f"peer{n}"))(i),
                          klass=Q.SPECULATIVE) for i in range(3)]
        release.set()
        q.drain(timeout=_WAIT)
        r.ok(f"a preempted job resumes ahead of later peers: {seen}") \
            if seen and seen[0] == "victim" else \
            r.fail("SCHED-4 head-requeue", f"{seen}")


def test_v031_sched4_real_cook_preemption(r: SubTestResult):
    """The row that stops every test above from passing vacuously: a REAL `tex_engine.cook`
    is preempted at a real SCHED-3 yield point, and the interactive frame that jumped the
    queue is bit-identical to the same cook run alone."""
    print("\n--- v0.31 SCHED-4: preempting a real tex_engine cook ---")
    from TEX_Wrangle import tex_engine

    A = make_img(1, 256, 256, 4, seed=31)
    # Many top-level statements => many interpreter yield points, so the preempt lands mid-cook.
    slow = "\n".join([f"float v{i} = luma(@A) * {1.0 + i * 0.01};" for i in range(200)]) + \
           "\n@OUT = vec4(vec3(v199), 1.0);"
    quick = "@OUT = vec4(@A.rgb * 1.25 + vec3(0.01), 1.0);"
    reference = tex_engine.cook(quick, {"A": A.clone()}, device_mode="cpu").outputs["OUT"]

    cooking = threading.Event()

    def spec_cook(cancel):
        return tex_engine.cook(slow, {"A": A.clone()}, device_mode="cpu",
                               cancel=cancel,
                               on_progress=lambda phase, frac: cooking.set())

    with Q.CookQueue() as q:
        spec = q.submit(spec_cook, klass=Q.SPECULATIVE, reason="panel-open")
        if not cooking.wait(_WAIT):
            r.fail("SCHED-4 real cook", "the speculative cook never began executing statements")
            return
        inter = q.submit(lambda cancel: tex_engine.cook(quick, {"A": A.clone()},
                                                       device_mode="cpu", cancel=cancel),
                         klass=Q.INTERACTIVE)
        try:
            got = inter.result(timeout=_WAIT).outputs["OUT"]
        except TimeoutError:
            r.fail("SCHED-4 real cook", "the interactive cook never ran")
            return
        md = (got.float() - reference.float()).abs().max().item()
        r.ok(f"the preempting interactive cook is bit-identical to a solo cook ({md:.1e})") \
            if md == 0.0 else r.fail("SCHED-4 real cook", f"maxdiff {md:.2e}")
        r.ok("the real cook was preempted at a SCHED-3 yield point") \
            if q.snapshot()["stats"]["preempted"] >= 1 else \
            r.fail("SCHED-4 real cook", "no preemption was recorded")

        # And the abandoned work is not lost — it either re-cooks (attempts ≥ 2) or the cook
        # returned before the preempt landed and is delivered anyway (§4b). Both are correct, and
        # which one happens depends on the starvation quantum against this program's speed, so
        # asserting `attempts >= 2` here would be asserting a race. That the requeue path works
        # is pinned by `test_v031_sched4_priority_and_preemption`, which controls the timing.
        try:
            res = spec.result(timeout=30.0)
        except (TimeoutError, CookCancelled) as e:
            r.fail("SCHED-4 real cook", f"the preempted cook was abandoned: {e!r}")
            return
        r.ok(f"the preempted cook is delivered, not lost (attempts={spec.attempts})") \
            if res.outputs["OUT"].shape == (1, 256, 256, 4) else \
            r.fail("SCHED-4 real cook", f"shape={tuple(res.outputs['OUT'].shape)}")


def test_v031_sched4_off_the_default_path(r: SubTestResult):
    """INVARIANT #7, as a canary rather than an assertion: the ComfyUI cook path must not
    import the queue, and constructing one must not start a thread until work arrives."""
    print("\n--- v0.31 SCHED-4: invariant #7 (the queue is not on the default path) ---")
    import re as _re
    # A SWEEP over the WHOLE package, not a name list: the canary exists to catch the module
    # that starts importing the queue *next*, which a hardcoded roster by definition cannot.
    # IMPORT forms only — the claim is "no engine module DEPENDS on the queue", and a prose
    # mention in a comment (profile.py explains which two threads its lock exists for) is not a
    # dependency; matching the bare token made this fire on documentation.
    offenders = lint_sources(
        r"^[ 	]*(?:from[ 	]+\.*[\w.]*tex_cookqueue|import[ 	]+[\w.]*tex_cookqueue)",
        allow={"tex_cookqueue.py"}, flags=_re.MULTILINE)
    r.ok("no engine or adapter module imports tex_cookqueue") if not offenders else         r.fail("SCHED-4 invariant #7", f"imported by {offenders}")

    before = threading.active_count()
    q = Q.CookQueue()
    idle = threading.active_count()
    q.submit(lambda cancel: 1, klass=Q.INTERACTIVE).result(timeout=_WAIT)
    q.close(timeout=_WAIT)
    r.ok("constructing a CookQueue starts no thread (the worker waits for work)") \
        if idle == before else \
        r.fail("SCHED-4 invariant #7", f"thread count {before} -> {idle} on construction")
