"""v0.34 IO-1 (async source materialization) + §3.3 (the async-write contract).

Design doc: `docs/async-io.md`. The rows follow its §7 definition of done.

Two mechanisms are genuinely new and everything else is templated, so the rows concentrate
there: a value-less BINDING in an engine that derives identity from binding values, and
dependency-aware ADMISSION in a queue whose jobs are opaque closures.
"""
import threading
import time

import torch

from helpers import devices as _devices
from TEX_Wrangle import tex_cookqueue as Q, tex_engine, tex_provider
from TEX_Wrangle.tex_compiler.types import TEXType
from TEX_Wrangle.tex_marshalling import Promise, infer_binding_type


def _img(res=8, dev="cpu"):
    return torch.rand(1, res, res, 4, device=dev)


# ── (a) the value-less binding ───────────────────────────────────────────────

def test_v034_io1_unknown_bindings_are_refused(r):
    """`infer_binding_type` refuses an unrecognised object as E7005.

    THE ROW WITH THE BLAST RADIUS. The old catch-all `return TEXType.FLOAT` meant a promise
    (or any object nobody thought about) typed as FLOAT, minted a fingerprint for a program
    that does not exist, and compiled something wrong — identity corruption with no error
    anywhere. Doc 41 is explicit this ships even if the rest of IO-1 slips."""
    try:
        try:
            infer_binding_type(object())
            r.fail("IO-1 loud refusal", "an unknown object still types as FLOAT")
            return
        except Exception as e:
            assert getattr(e, "_code", "") == "E7005", f"{getattr(e, '_code', '')}: {e}"

        # Every type the tree relies on keeps its branch — this is the half that says the
        # refusal did not cost anything real.
        assert infer_binding_type(torch.rand(1, 4, 4, 3)) is TEXType.VEC3
        assert infer_binding_type(torch.rand(1, 4, 4)) is TEXType.FLOAT
        assert infer_binding_type(1) is TEXType.INT
        assert infer_binding_type(True) is TEXType.INT
        assert infer_binding_type(1.5) is TEXType.FLOAT
        assert infer_binding_type("s") is TEXType.STRING
        assert infer_binding_type([torch.rand(1, 4, 4, 4)]) is TEXType.VEC4
        # None is an UNCONNECTED optional input: the E6003 gate reports it further down,
        # naming the slot, which this function cannot.
        assert infer_binding_type(None) is TEXType.FLOAT
        r.ok("IO-1: an unknown binding is E7005; every relied-on type keeps its branch")
    except Exception as e:
        r.fail("IO-1 loud refusal", f"{type(e).__name__}: {e}")


def test_v034_io1_a_promise_declares_its_identity(r):
    """Identity is computable before the pixels land, and the landing is validated."""
    try:
        p = Promise("A", type=TEXType.VEC4, shape=(1, 8, 8, 4), device="cpu")
        assert infer_binding_type(p) is TEXType.VEC4, "a promise did not type by declaration"
        assert not p.landed

        # Declared shape omitted for the type case on purpose: the shape check runs first,
        # so a wrong-CHANNEL tensor would report the shape mismatch and this row would be
        # asserting the same branch twice.
        for label, bad, why, shape in (
                ("shape", torch.rand(1, 4, 4, 4), "declared shape", (1, 8, 8, 4)),
                ("type", torch.rand(1, 8, 8, 3), "declared VEC4", None)):
            q = Promise("A", type=TEXType.VEC4, shape=shape)
            try:
                q.land(bad)
                r.fail("IO-1 promise validation", f"a wrong-{label} landing was accepted")
                return
            except Exception as e:
                assert getattr(e, "_code", "") == "E7006", f"{label}: {e}"
                assert why in str(e), str(e)
                assert not q.landed, f"a refused landing left {label} promise landed"

        t = _img()
        p.land(t)
        assert p.landed and p.value is t
        r.ok("IO-1: a promise types by declaration and validates its landing (E7006)")
    except Exception as e:
        r.fail("IO-1 promise identity", f"{type(e).__name__}: {e}")


def test_v034_io1_prepare_resolves_and_refuses(r):
    """A landed promise cooks as its tensor; an unlanded one is E7007, not a wait.

    Refusing rather than waiting is the point: blocking the single cook worker on host I/O
    is exactly what the WAITING state exists to prevent, and a stall is harder to diagnose
    than an error."""
    for dev in _devices():
        try:
            src = _img(dev=dev)
            p = Promise("A", type=TEXType.VEC4, shape=tuple(src.shape), device=str(src.device))
            p.land(src)
            got = tex_engine.cook("@OUT = vec4(@A.rgb * 2.0, 1.0);", {"A": p},
                                  device_mode=dev).outputs["OUT"]
            direct = tex_engine.cook("@OUT = vec4(@A.rgb * 2.0, 1.0);", {"A": src},
                                     device_mode=dev).outputs["OUT"]
            assert torch.equal(got, direct), "cooking through a promise moved pixels"

            unlanded = Promise("A", type=TEXType.VEC4)
            try:
                tex_engine.cook("@OUT = vec4(@A.rgb, 1.0);", {"A": unlanded}, device_mode=dev)
                r.fail("IO-1 unlanded refusal", "an unlanded promise cooked")
                return
            except Exception as e:
                assert getattr(e, "_code", "") == "E7007", f"{getattr(e, '_code', '')}: {e}"
            r.ok(f"IO-1: a landed promise cooks bit-identically; an unlanded one is E7007 ({dev})")
        except Exception as e:
            r.fail(f"IO-1 prepare resolution ({dev})", f"{type(e).__name__}: {e}")


# ── (b) dependency-aware admission ───────────────────────────────────────────

def test_v034_io1_a_ready_branch_cooks_while_its_sibling_waits(r):
    """THE ITEM'S HEADLINE, asserted by cook ORDER rather than by timing.

    Two jobs submitted in order: the first waits on an unlanded promise, the second does not.
    The second must run first. Per-branch granularity falls out with no extra work because
    the unfused host already submits per-stage jobs."""
    q = Q.CookQueue(name="io1-order")
    order = []
    try:
        p = Promise("A", type=TEXType.VEC4)
        waiter = q.submit(lambda c: order.append("waiter"), klass=Q.COMMITTED, inputs=[p])
        ready = q.submit(lambda c: order.append("ready"), klass=Q.COMMITTED)
        assert ready.wait(10), "the ready job never finished"
        assert order == ["ready"], f"order={order} — the waiter ran before its input landed"
        assert waiter.state == Q.WAITING, waiter.state
        assert q.stats.waiting == 1, q.stats.waiting
        snap = q.snapshot()
        assert snap["waiting"]["committed"] == 1, snap

        p.land(_img())
        assert waiter.wait(10), "the waiter never woke after its input landed"
        assert order == ["ready", "waiter"], order
        assert q.stats.waiting == 0, q.stats.waiting
        r.ok("IO-1: a ready branch cooks while its sibling waits; landing wakes the waiter")
    except Exception as e:
        r.fail("IO-1 waiting admission", f"{type(e).__name__}: {e}")
    finally:
        q.close()


def test_v034_io1_a_waiting_submit_never_preempts(r):
    """A WAITING submit must not trip a running job.

    A preempted cook loses ALL its progress (§4a — there is no resume), so preempting for a
    job that cannot start is the worst trade available: the render restarts and the
    preemptor is still waiting on a disk read. The preemption it was denied is granted when
    the input lands, which the second half asserts."""
    q = Q.CookQueue(name="io1-preempt", min_quantum_ms=0.0)
    started, release = threading.Event(), threading.Event()
    try:
        def slow(cancel):
            started.set()
            for _ in range(400):
                cancel.check()
                if release.wait(0.005):
                    break
            return "slow"

        low = q.submit(slow, klass=Q.COMMITTED)
        assert started.wait(5), "the long job never started"
        p = Promise("A", type=TEXType.VEC4)
        q.submit(lambda c: "waiter", klass=Q.INTERACTIVE, inputs=[p])
        time.sleep(0.1)
        assert q.stats.preempted == 0, \
            f"a WAITING interactive submit preempted a running job ({q.stats.preempted})"

        p.land(_img())                       # now it CAN run — and now it may preempt
        deadline = time.perf_counter() + 5
        while q.stats.preempted == 0 and time.perf_counter() < deadline:
            time.sleep(0.01)
        assert q.stats.preempted >= 1, "landing did not grant the deferred preemption"
        release.set()
        low.wait(10)
        r.ok("IO-1: a waiting submit does not preempt; landing grants the deferred preemption")
    except Exception as e:
        r.fail("IO-1 waiting preemption", f"{type(e).__name__}: {e}")
    finally:
        release.set()
        q.close()


def test_v034_io1_a_failed_promise_fails_its_jobs(r):
    """A source that cannot be read is a finished question, not a slow one.

    Without this a job parks forever and the queue is indistinguishable from idle — the
    failure mode `stats.waiting` exists to make visible."""
    q = Q.CookQueue(name="io1-fail")
    try:
        p = Promise("A", type=TEXType.VEC4)
        job = q.submit(lambda c: "never", klass=Q.COMMITTED, inputs=[p])
        boom = OSError("the plate is gone")
        p.fail(boom)
        assert job.wait(10), "a failed promise left its job parked"
        assert job.state == Q.FAILED, job.state
        assert job.error is boom, job.error
        assert q.stats.waiting == 0, q.stats.waiting
        r.ok("IO-1: a failed promise fails its jobs with the host's own exception")
    except Exception as e:
        r.fail("IO-1 promise failure", f"{type(e).__name__}: {e}")
    finally:
        q.close()


def test_v034_io1_waiting_jobs_stay_visible_to_shed_and_close(r):
    """A WAITING job lives in its class deque, so every existing path still sees it.

    This is why the design refused a side table: the shed policy is precisely what must be
    able to drop a waiting speculative prefetch, and `close()` must not leak one. The
    `stats.waiting` counter is decremented through `_finish_locked`, the one door all four
    finishing paths go through, which is why it cannot drift."""
    q = Q.CookQueue(name="io1-shed")
    try:
        q.install_policy(Q.SpeculativePolicy(min_confidence=0.0, min_value_ms=0.0,
                                             unknown_min_confidence=0.0, max_pending=2))
        promises = [Promise(f"P{i}", type=TEXType.VEC4) for i in range(5)]
        jobs = [q.submit(lambda c: None, klass=Q.SPECULATIVE, reason=Q.PREFETCH,
                         confidence=0.5, inputs=[p]) for p in promises]
        time.sleep(0.1)
        shed = [j for j in jobs if j.state == Q.CANCELLED]
        assert shed, "the shed policy could not see the WAITING prefetches"
        assert q.stats.waiting == sum(1 for j in jobs if j.state == Q.WAITING), q.stats.waiting

        q.close()
        assert all(j.state in (Q.CANCELLED, Q.DONE, Q.FAILED) for j in jobs), \
            [j.state for j in jobs]
        assert q.stats.waiting == 0, f"close() leaked {q.stats.waiting} waiting jobs"
        r.ok("IO-1: waiting jobs are sheddable and closable; the waiting counter never drifts")
    except Exception as e:
        r.fail("IO-1 waiting visibility", f"{type(e).__name__}: {e}")
    finally:
        q.close()


# ── prefetch, the pollution guard, cancellation, backpressure ────────────────

def test_v034_io1_prefetch_never_feeds_the_profiler(r):
    """PROF-1 pollution guard, structural rather than incidental.

    A prefetch is I/O-bound; its wall time recorded as compute cost poisons CACHE-7
    placement and PRED-1 admission in one stroke. The guard is an explicit `feeds_profile`
    flag, not "we happened not to pass a profile key" — the host controls that argument, so
    this row submits a slow prefetch WITH one and asserts the table is unchanged."""
    from TEX_Wrangle.tex_runtime import profile as P
    q = Q.CookQueue(name="io1-prof")
    try:
        key = P.make_key("io1-guard-fp", "cpu", "fp32")
        before = P.predict(key, 1024)
        job = q.submit(lambda c: time.sleep(0.05) or "slow", klass=Q.SPECULATIVE,
                       reason=Q.PREFETCH, confidence=1.0, cost_ms=1.0,
                       profile_key=key, px=1024, feeds_profile=False)
        job.wait(10)
        assert job.state == Q.DONE, job.state
        assert P.predict(key, 1024) == before, \
            f"a prefetch fed PROF-1: {before} -> {P.predict(key, 1024)}"

        # The negative control: the same job WITHOUT the flag does feed it, so the row is
        # not passing because the feedback is broken.
        job2 = q.submit(lambda c: time.sleep(0.05) or "slow", klass=Q.COMMITTED,
                        profile_key=key, px=1024)
        job2.wait(10)
        assert P.predict(key, 1024) is not None, "the profiler feedback itself is dead"
        r.ok("IO-1: feeds_profile=False keeps a prefetch out of PROF-1 (control: it feeds)")
    except Exception as e:
        r.fail("IO-1 PROF-1 guard", f"{type(e).__name__}: {e}")
    finally:
        q.close()


def test_v034_io1_declare_window_prefetches_a_range(r):
    """`declare_window` mints one SPECULATIVE job per quantized frame, and they populate
    the pool without any cook asking for them."""
    q = Q.CookQueue(name="io1-window")
    tex_provider.reset_provider()
    p = tex_provider.SyntheticFrameProvider(res=16, rate=1.0)
    tex_provider.set_provider(p)
    try:
        q.install_policy(Q.SpeculativePolicy(min_confidence=0.0, min_value_ms=0.0,
                                             unknown_min_confidence=0.0, max_pending=16))
        jobs = tex_provider.declare_window(q, "plate", 0.0, 3.0, confidence=0.9)
        assert len(jobs) == 4, f"{len(jobs)} jobs for a 4-frame window"
        q.drain(10)
        st = tex_provider.get_media_cache().stats()
        assert st["frames"] == 4, st
        assert p.fetches == 4, p.fetches
        assert all(j.feeds_profile is False for j in jobs), "a prefetch job feeds PROF-1"
        r.ok("IO-1: declare_window prefetches a whole range into the pool, off Tier-B")
    except Exception as e:
        r.fail("IO-1 prefetch window", f"{type(e).__name__}: {e}")
    finally:
        q.close()
        tex_provider.reset_provider()


def test_v034_io1_backpressure_refuses_rather_than_evicting(r):
    """A speculative insert into a full pool is REFUSED, not admitted-then-evicting.

    A prefetch is a guess; evicting a frame someone actually asked for to make room for a
    guess inverts the whole policy. Refusal is cheap because the frame is still on disk."""
    tex_provider.reset_provider()
    tex_provider.set_provider(tex_provider.SyntheticFrameProvider(res=64, rate=1.0))
    try:
        # One demanded frame, then a budget that leaves no room for a guess.
        tex_provider.materialize("plate", 0.0, "fetch")
        cache = tex_provider.get_media_cache()
        held = cache.stats()["bytes"]
        tex_provider.set_media_budget_mb(held / (1024 * 1024))

        got = tex_provider.materialize("plate", 1.0, "fetch", speculative=True)
        st = cache.stats()
        assert got is None, "a speculative insert into a full pool was admitted"
        assert st["refused"] == 1, st
        assert st["frames"] == 1, f"the demanded frame was evicted for a guess: {st}"

        # ...and a DEMANDED read still gets in, evicting as usual. Backpressure applies to
        # guesses only; a frame the user asked for is never refused.
        assert tex_provider.materialize("plate", 2.0, "fetch") is not None
        r.ok("IO-1: backpressure refuses a speculative insert but never a demanded one")
    except Exception as e:
        r.fail("IO-1 backpressure", f"{type(e).__name__}: {e}")
    finally:
        tex_provider.set_media_budget_mb(512.0)
        tex_provider.reset_provider()


def test_v034_io1_cancellation_drops_on_landing(r):
    """A cancelled prefetch's result is never installed.

    The provider's read cannot be stopped from outside once it has begun; what TEX
    guarantees is that nothing the host did not still want ends up in the pool."""
    q = Q.CookQueue(name="io1-cancel")
    tex_provider.reset_provider()
    tex_provider.set_provider(tex_provider.SyntheticFrameProvider(res=16, rate=1.0,
                                                                  latency_s=0.05))
    try:
        q.install_policy(Q.SpeculativePolicy(min_confidence=0.0, min_value_ms=0.0,
                                             unknown_min_confidence=0.0, max_pending=16))
        jobs = tex_provider.declare_window(q, "plate", 0.0, 5.0, confidence=0.9)
        # Cancel the tail while the head is still inside the provider's sleep.
        for j in jobs[2:]:
            q.cancel(j)
        q.drain(20)
        cancelled = sum(1 for j in jobs if j.state == Q.CANCELLED)
        assert cancelled >= 1, [j.state for j in jobs]
        frames = tex_provider.get_media_cache().stats()["frames"]
        assert frames <= len(jobs) - cancelled + 1, \
            f"{frames} frames pooled after {cancelled} cancellations of {len(jobs)}"
        r.ok(f"IO-1: {cancelled} cancelled prefetches installed nothing (pool={frames})")
    except Exception as e:
        r.fail("IO-1 cancellation", f"{type(e).__name__}: {e}")
    finally:
        q.close()
        tex_provider.reset_provider()


# ── §3.3 the async-write contract ────────────────────────────────────────────

def test_v034_async_write_does_not_block_the_next_cook(r):
    """THE SCHEDULING HALF. Frame N's handle goes to a slow writer; frame N+1 must start
    while that writer still holds it.

    TEX's obligation ends exactly here — at "the next cook starts immediately". Asserted by
    ORDER (the writer is still holding when N+1 executes), not by a timing threshold."""
    from TEX_Wrangle.tex_runtime import streams
    q = Q.CookQueue(name="io1-write")
    holding, n1_started = threading.Event(), threading.Event()
    release = threading.Event()
    try:
        frame_n = tex_engine.cook("@OUT = vec4(@A.rgb, 1.0);", {"A": _img(32)},
                                  device_mode="cpu").outputs["OUT"]
        handle = streams.egress(frame_n)

        written = {}

        def slow_writer():
            holding.set()
            release.wait(10)
            written["bytes"] = handle.tensor().clone()   # fences, then "writes"

        t = threading.Thread(target=slow_writer, daemon=True)
        t.start()
        assert holding.wait(5), "the writer never started"

        def cook_n1(cancel):
            n1_started.set()
            return tex_engine.cook("@OUT = vec4(@A.rgb * 0.5, 1.0);", {"A": _img(32)},
                                   device_mode="cpu", cancel=cancel)

        job = q.submit(cook_n1, klass=Q.COMMITTED)
        assert n1_started.wait(10), "frame N+1 never started while the writer held N"
        assert not written, "the writer finished early — the test proved nothing"
        assert job.wait(20) and job.state == Q.DONE, job.state

        release.set()
        t.join(10)
        assert torch.equal(written["bytes"], frame_n.cpu()), \
            "the async-written bytes differ from the source frame"
        r.ok("async writes: frame N+1 cooks and completes while the writer still holds N")
    except Exception as e:
        r.fail("async write scheduling", f"{type(e).__name__}: {e}")
    finally:
        release.set()
        q.close()


def test_v034_async_write_bytes_are_bit_exact(r):
    """THE FENCE HALF. What a handle delivers equals a synchronous read, byte for byte,
    on every device — and the two reference consumers accept a handle."""
    from TEX_Wrangle.tex_runtime import streams
    for dev in _devices():
        try:
            src = _img(64, dev=dev)
            sync = src.detach().float().cpu().clone()
            got = streams.egress(src).tensor()
            assert torch.equal(got, sync), "a fenced handle's bytes differ from a sync copy"

            # host_demo's blit is the shipped consumer: handle and tensor must agree.
            import importlib.util
            import os
            hd_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "examples", "host_demo.py")
            spec = importlib.util.spec_from_file_location("tex_host_demo_probe", hd_path)
            hd = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(hd)
            host = hd.Host.__new__(hd.Host)          # no server, no cook: only the blit
            assert host.rgba_bytes(streams.egress(src)) == host.rgba_bytes(src), \
                "host_demo's blit disagrees between a handle and a tensor"
            r.ok(f"async writes: handle bytes are bit-exact and the blit accepts both ({dev})")
        except Exception as e:
            r.fail(f"async write fence ({dev})", f"{type(e).__name__}: {e}")
