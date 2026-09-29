"""v0.52 sweep: cook-queue ordering, admission and lifecycle rows.

Each row failed on the base it was written against: shed ties, a failed promise at submit,
a submit that raises after enqueueing, a preemption whose requester was cancelled, retained
finished handles, and NaN confidence.
"""
import threading
import time

import pytest

from TEX_Wrangle import tex_cookqueue as Q

_WAIT = 10.0


def _hold(q, klass=Q.INTERACTIVE):
    """Park the worker on a gate so later submits stay queued. Returns (gate, job)."""
    gate, started = threading.Event(), threading.Event()

    def fn(cancel):
        started.set()
        gate.wait(_WAIT)

    job = q.submit(fn, klass=klass)
    assert started.wait(_WAIT)
    return gate, job


def test_shed_tie_evicts_the_newest_speculative_job():
    with Q.CookQueue() as q:
        gate, _ = _hold(q)
        jobs = [q.submit(lambda c: None, klass=Q.SPECULATIVE) for _ in range(4)]
        q.shed_speculative(keep=2)
        assert [j.state for j in jobs] == [Q.PENDING, Q.PENDING, Q.CANCELLED, Q.CANCELLED]
        gate.set()
        q.drain(timeout=_WAIT)


def test_shed_tie_protects_a_resumed_job():
    started, release = threading.Event(), threading.Event()

    def blocker(cancel):
        started.set()
        while not release.wait(0.002):
            cancel.check()
        return "done"

    with Q.CookQueue(min_quantum_ms=0.0) as q:
        first = q.submit(blocker, klass=Q.SPECULATIVE)
        assert started.wait(_WAIT)
        hold = threading.Event()
        q.submit(lambda c: hold.wait(_WAIT), klass=Q.INTERACTIVE)
        for _ in range(int(_WAIT / 0.002)):
            if q.snapshot()["stats"]["requeued"] >= 1:
                break
            time.sleep(0.002)
        assert q.snapshot()["stats"]["requeued"] == 1 and first.resumed
        later = q.submit(lambda c: None, klass=Q.SPECULATIVE)
        q.shed_speculative(keep=1)
        assert first.state == Q.PENDING and later.state == Q.CANCELLED
        release.set()
        hold.set()
        q.drain(timeout=_WAIT)


class _FailedPromise:
    landed = True

    def __init__(self, err):
        self.error = err

    def on_land(self, cb):
        cb(self)


def test_failed_input_at_submit_fails_the_job_without_preempting():
    err = RuntimeError("source gone")
    ran = []
    with Q.CookQueue(min_quantum_ms=0.0) as q:
        gate, run = _hold(q, klass=Q.COMMITTED)
        job = q.submit(lambda c: ran.append(1), klass=Q.INTERACTIVE,
                       inputs=[_FailedPromise(err)])
        assert job.state == Q.FAILED and job.error is err
        assert not run.preempt_requested
        st = q.snapshot()["stats"]
        assert st["preempted"] == 0 and st["waiting"] == 0 and st["failed"] == 1
        gate.set()
        q.drain(timeout=_WAIT)
        assert run.state == Q.DONE and run.preemptions == 0 and not ran


def test_submit_that_cannot_start_a_worker_leaves_no_orphan_job():
    ran = []
    q = Q.CookQueue()
    try:
        real = q._ensure_worker

        def boom():
            raise RuntimeError("can't start new thread")

        q._ensure_worker = boom
        with pytest.raises(RuntimeError):
            q.submit(lambda c: ran.append("orphan"))
        assert not any(q._q[k] for k in Q.CLASSES)
        q._ensure_worker = real
        q.submit(lambda c: ran.append("ok")).wait(_WAIT)
        assert ran == ["ok"]
    finally:
        q.close()


def test_cancelling_the_preemptor_retracts_the_preempt_request():
    with Q.CookQueue(min_quantum_ms=0.0) as q:
        gate, run = _hold(q, klass=Q.COMMITTED)
        bump = q.submit(lambda c: None, klass=Q.INTERACTIVE)
        assert run.preempt_requested
        assert q.cancel(bump)
        assert not run.preempt_requested
        assert q.snapshot()["stats"]["preempted"] == 0
        # A second interactive job still queued keeps the request alive.
        a = q.submit(lambda c: None, klass=Q.INTERACTIVE)
        b = q.submit(lambda c: None, klass=Q.INTERACTIVE)
        assert run.preempt_requested
        q.cancel(a)
        assert run.preempt_requested
        q.cancel(b)
        assert not run.preempt_requested
        gate.set()
        q.drain(timeout=_WAIT)
        assert run.preemptions == 0


def test_finished_job_releases_its_closure_and_inputs():
    with Q.CookQueue() as q:
        big = bytearray(10)
        job = q.submit(lambda c: len(big))
        assert job.result(_WAIT) == 10
        assert job.fn is None and job.inputs == ()

        def bad(c):
            raise ValueError("x")
        failed = q.submit(bad)
        failed.wait(_WAIT)
        assert failed.state == Q.FAILED and failed.fn is None
        assert failed.error.__traceback__ is None


def test_nan_confidence_is_refused_not_certain():
    with Q.CookQueue() as q:
        q.install_policy(Q.SpeculativePolicy())
        gate, _ = _hold(q)
        job = q.submit(lambda c: None, klass=Q.SPECULATIVE, confidence=float("nan"),
                       cost_ms=100.0)
        assert job.state == Q.CANCELLED and q.snapshot()["stats"]["refused"] == 1
        gate.set()
        q.drain(timeout=_WAIT)
