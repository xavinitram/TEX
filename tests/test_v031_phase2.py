"""v0.31 Phase 2 — PROF-1 (the cost profiler) and PRED-1 (the speculative-cook protocol).

PROF-1 is the number "effort-based" has been missing: the engine's only shipped cost signal is
`autotier.cook_ms`, which is whole-program AND only ever fed on the `compile_mode="auto"` path
— the default interpreter cook records nothing. What has to be pinned:

  * DISARMED BY DEFAULT (invariant #7 applies to the profiler itself — doc 39 §8).
  * The sampling gate is a real rate limiter, not decoration: warmup cooks always measure,
    then one in `_SAMPLE_EVERY`. It is what makes the CUDA sync affordable.
  * The per-STAGE breakdown attributes cost to the right stage of a FUSED chain. Pinned with
    a chain whose stage 1 is deliberately ~10x the others, so a test that merely divided the
    total evenly would fail.
  * A cook that RAISED is not recorded — an OOM or a cancel took an unrepresentative time.

PRED-1 is the economics on top. The pinned facts are the two halves of `confidence × cost`,
and the second is the counter-intuitive one:

  * a job nobody will probably want is refused however expensive;
  * a job everyone will want is refused if it is CHEAP — there is no latency to hide.

Shapes (roadmap §10.4): CANARY for the disarmed default and the class exemptions, DERIVATION
for the admission arithmetic (explicit numbers in → explicit verdict out, autotier's
discipline), NEVER-SEVER ROWS for the shed order.
"""
import contextlib
import threading

from helpers import *

from TEX_Wrangle import tex_engine, tex_cookqueue as Q
from TEX_Wrangle.tex_runtime import profile as P

_WAIT = 10.0


@contextlib.contextmanager
def _armed():
    """Arm the profiler on a clean table, and always disarm — a leaked `enable()` would put a
    CUDA sync into every later test in the suite."""
    P.reset()
    P.enable()
    try:
        yield P
    finally:
        P.disable()
        P.reset()


# ── PROF-1 ────────────────────────────────────────────────────────────────────

def test_v031_prof1_disarmed_by_default(r: SubTestResult):
    print("\n--- v0.31 PROF-1: disarmed by default (invariant #7) ---")
    P.reset()
    r.ok("the profiler is disabled at import") if not P.enabled() else \
        r.fail("PROF-1 default", "enabled() is True with nothing armed")

    A = make_img(1, 32, 32, 4, seed=31)
    tex_engine.cook("@OUT = vec4(@A.rgb * 1.1, 1.0);", {"A": A}, device_mode="cpu")
    r.ok("a default cook records nothing") if not P.snapshot() else \
        r.fail("PROF-1 default", f"the disarmed profiler recorded {len(P.snapshot())} keys")

    key = P.make_key("nope", "cpu", "fp32")
    r.ok("should_sample() is False while disarmed (the only gate a caller needs)") \
        if not P.should_sample(key, 1024) else \
        r.fail("PROF-1 default", "should_sample() said yes while disarmed")


def test_v031_prof1_records_and_predicts(r: SubTestResult):
    print("\n--- v0.31 PROF-1: an armed cook is measured and predicted ---")
    with _armed():
        A = make_img(1, 128, 128, 4, seed=31)
        code = "@OUT = vec4(@A.rgb * 1.1 + vec3(0.01), 1.0);"
        for _ in range(4):
            tex_engine.cook(code, {"A": A}, device_mode="cpu")
        snap = P.snapshot()
        r.ok(f"an armed cook records a cost ({len(snap)} key(s))") if snap else \
            r.fail("PROF-1 record", "an armed cook recorded nothing")

        # The key the engine used, rebuilt the way a host would.
        from TEX_Wrangle.tex_cache import get_cache
        from TEX_Wrangle.tex_compiler.types import TEXType
        fp = get_cache().fingerprint(code, {"A": TEXType.VEC4})
        key = P.make_key(fp, "cpu", "fp32")
        ms = P.predict(key, 128 * 128)
        r.ok(f"predict() answers for a measured program ({ms:.3f} ms, "
             f"{P.samples(key, 128 * 128)} samples)") \
            if ms is not None and ms > 0 else \
            r.fail("PROF-1 predict", f"predict returned {ms!r} for a key that was just cooked")

        r.ok("predict() is None for a program never cooked") \
            if P.predict(P.make_key("never-seen", "cpu", "fp32"), 1024) is None else \
            r.fail("PROF-1 predict", "an unmeasured program returned a cost")


def test_v031_prof1_sampling_gate(r: SubTestResult):
    """The rate limiter is the reason a CUDA-syncing profiler can be left armed at all."""
    print("\n--- v0.31 PROF-1: warmup then 1-in-N sampling ---")
    with _armed():
        key = P.make_key("gate-probe", "cpu", "fp32")
        px = 512 * 512
        warm = [P.should_sample(key, px) for _ in range(P._WARMUP_SAMPLES)]
        r.ok(f"the first {P._WARMUP_SAMPLES} cooks of an unseen key all sample") \
            if all(warm) else r.fail("PROF-1 gate", f"warmup pattern {warm}")

        # An unseen key samples every time until it has real samples — feed them.
        for _ in range(P._WARMUP_SAMPLES):
            P.record(key, 5.0, px)
        after = [P.should_sample(key, px) for _ in range(P._SAMPLE_EVERY * 3)]
        hits = sum(after)
        r.ok(f"after warmup: {hits} samples in {len(after)} cooks (1 in {P._SAMPLE_EVERY})") \
            if hits == 3 else \
            r.fail("PROF-1 gate", f"{hits} samples in {len(after)} cooks, expected 3")


def test_v031_prof1_per_stage_breakdown(r: SubTestResult):
    """CACHE-7's input. A three-stage fused chain where stage 1 is four chained blurs and the
    other two are single pointwise ops: the breakdown has to FINGER STAGE 1, or 'effort-based
    placement' would cut the chain in the wrong place.

    What is asserted is that stage 1 is the clear maximum, not that it exceeds the sum of the
    others — a cut point is chosen by comparing boundaries to each other, and the two cheap
    stages carry a fixed per-cook floor (binding setup, the first statement's lazy init) that
    a ratio test would keep tripping over at small resolutions."""
    print("\n--- v0.31 PROF-1: per-stage cost in a fused chain ---")
    with _armed():
        A = make_img(1, 256, 256, 4, seed=31)
        heavy_code = "@OUT = gauss_blur(gauss_blur(gauss_blur(gauss_blur(@IN, 8.0), 8.0), 8.0), 8.0);"
        payload = {"schema": 1,
                   "stages": [{"code": "@OUT = vec4(@IN.rgb * 1.15, 1.0);",
                               "image_input": "IN", "params": {}},
                              {"code": heavy_code, "image_input": "IN", "params": {}}],
                   "terminal_image_input": "IN"}
        term = "@OUT = vec4(@IN.rgb + vec3(0.01), 1.0);"
        # Warm the ENGINE first, then start the profiler's own table clean. The first cook of
        # a program is cold by construction (compile, allocator growth, first-touch of every
        # cached kernel) and lands ~10x its steady state; measuring only that would be
        # measuring the compiler. `_blend`'s running-mean warmup dilutes a cold sample fast,
        # but a test whose entire sample set is cold is testing the wrong thing.
        for _ in range(3):
            tex_engine.cook(term, {"IN": A}, chain_payload=payload, device_mode="cpu")
        P.reset()
        for _ in range(3):
            tex_engine.cook(term, {"IN": A}, chain_payload=payload, device_mode="cpu")

        snap = P.snapshot()
        stages = {}
        for buckets in snap.values():
            for b in buckets.values():
                if b["stages"]:
                    stages = b["stages"]
        if len(stages) < 3:
            r.fail("PROF-1 stages", f"expected a 3-stage breakdown, got {stages}")
            return
        r.ok(f"a fused chain profiles per stage: {stages}")
        ranked = sorted(stages.items(), key=lambda kv: -kv[1])
        (heavy, hms), (_second, sms) = ranked[0], ranked[1]
        r.ok(f"the four-blur stage is the clear maximum: stage {heavy} at {hms:.2f} ms vs "
             f"{sms:.2f} ms next") if heavy == "1" and hms > 2.0 * sms else \
            r.fail("PROF-1 stages",
                   f"the four-blur stage did not stand out: {stages}")


def test_v031_prof1_predicts_an_unseen_resolution(r: SubTestResult):
    print("\n--- v0.31 PROF-1: an unmeasured resolution scales from a measured one ---")
    with _armed():
        key = P.make_key("scale-probe", "cpu", "fp32")
        P.record(key, 10.0, 256 * 256)
        got = P.predict(key, 512 * 512)
        # 4x the pixels from the only measured bucket. Linear-in-pixels is the documented
        # over-approximation; what is pinned here is that it EXTRAPOLATES rather than
        # returning None (which would make a cold PRED-1 refuse everything forever).
        r.ok(f"4x the pixels predicts {got:.1f} ms from a 10.0 ms sample") \
            if got is not None and abs(got - 40.0) < 1e-6 else \
            r.fail("PROF-1 scale", f"predicted {got!r}, expected 40.0")


def test_v031_prof1_ignores_a_failed_cook(r: SubTestResult):
    print("\n--- v0.31 PROF-1: a cook that raised is not recorded ---")
    with _armed():
        key = P.make_key("raise-probe", "cpu", "fp32")
        try:
            with P.measure(key, 4096, device="cpu"):
                raise RuntimeError("simulated OOM")
        except RuntimeError:
            pass
        r.ok("an exception inside measure() records nothing") \
            if P.predict(key, 4096) is None else \
            r.fail("PROF-1 failure", "a raised cook poisoned the EWMA")


# ── PRED-1 ────────────────────────────────────────────────────────────────────

def _pol(**kw):
    return Q.SpeculativePolicy(**kw)


def test_v031_pred1_admission_arithmetic(r: SubTestResult):
    """DERIVATION: explicit numbers in, explicit verdict out. Both halves of `confidence x cost`
    are load-bearing, and the cheap-but-certain refusal is the one that looks wrong until you
    remember what speculation is for."""
    print("\n--- v0.31 PRED-1: admission by confidence x predicted cost ---")
    # A 10 ms floor, so the rows straddle it rather than all clearing the 2 ms default.
    _FLOOR = 10.0
    rows = [
        # (confidence, cost_ms, why this row exists)
        (0.9, 40.0, "likely and expensive — the case speculation exists for"),
        (0.02, 400.0, "a long render nobody will probably want (0.02 x 400 = 8)"),
        (1.0, 0.3, "certain but trivial — there is no latency to hide"),
        (0.4, 30.0, "12 ms of expected saving — over the floor on both terms"),
    ]
    policy = _pol(min_value_ms=_FLOOR, max_pending=99)
    for conf, cost, why in rows:
        job = Q.Job(id=0, klass=Q.SPECULATIVE, fn=lambda c: None,
                    confidence=conf, cost_ms=cost)
        got = policy.admit(job)
        expect = (conf * cost) >= _FLOOR
        r.ok(f"conf {conf} x {cost} ms = {job.score:.1f} -> "
             f"{'admit' if got else 'refuse'}  ({why})") if got == expect else \
            r.fail("PRED-1 arithmetic",
                   f"conf={conf} cost={cost} score={job.score} admitted={got}, expected {expect}")

    # And the default floor refuses the cheap-but-certain job while admitting the same
    # confidence on real work — the pair is the point.
    d = _pol()
    cheap = Q.Job(id=1, klass=Q.SPECULATIVE, fn=lambda c: None, confidence=1.0, cost_ms=0.3)
    real = Q.Job(id=2, klass=Q.SPECULATIVE, fn=lambda c: None, confidence=1.0, cost_ms=30.0)
    r.ok("at the default floor: a 0.3 ms certainty is refused, a 30 ms one admitted") \
        if not d.admit(cheap) and d.admit(real) else \
        r.fail("PRED-1 floor", f"cheap={d.admit(cheap)} real={d.admit(real)}")


def test_v031_pred1_bounds_each_factor_not_just_the_product(r: SubTestResult):
    """`confidence × cost` RANKS bets; it does not admit them. With neither factor bounded the
    policy admits the CHANGELOG's own counter-example verbatim — confidence 0.02 on a 400 ms
    render scores 8.0, clears any sane saving floor, and spends 400 ms of worker time on a 2%
    chance. So each factor carries its own bound."""
    print("\n--- v0.31 PRED-1: the product rule needs per-factor bounds ---")
    policy = _pol()          # shipped defaults

    unlikely = Q.Job(id=1, klass=Q.SPECULATIVE, fn=lambda c: None,
                     confidence=0.02, cost_ms=400.0, reason=Q.PREFETCH)
    r.ok(f"conf 0.02 x 400 ms (score {policy.score_of(unlikely):.1f}) is REFUSED on confidence") \
        if not policy.admit(unlikely) else \
        r.fail("PRED-1 bounds", "a 2%-likely 400 ms render was admitted")

    huge = Q.Job(id=2, klass=Q.SPECULATIVE, fn=lambda c: None,
                 confidence=0.99, cost_ms=30_000.0, reason=Q.PANEL_OPEN)
    r.ok("a 30 s cook is REFUSED however likely (submit it as COMMITTED)") \
        if not policy.admit(huge) else \
        r.fail("PRED-1 bounds", "a 30 s speculative cook was admitted")

    good = Q.Job(id=3, klass=Q.SPECULATIVE, fn=lambda c: None,
                 confidence=0.6, cost_ms=40.0, reason=Q.PLAY_HOVER)
    r.ok("a likely, sensibly-sized bet is still admitted") if policy.admit(good) else \
        r.fail("PRED-1 bounds", "the bounds refuse the case speculation exists for")

    r.ok(f"each refusal is attributed: {sorted(policy.refusals)}") \
        if len(policy.refusals) == 2 else \
        r.fail("PRED-1 bounds", f"{policy.refusals}")


def test_v031_prof1_fused_chains_key_apart(r: SubTestResult):
    """PROF-1's key is `fused_fp or fp`. On the DEFAULT `compile_mode="none"` a fused chain used
    to have neither — `fused_fp` was only computed for the compiling modes and `fp` is None on a
    chain — so the key degenerated to `None|device|precision` and two structurally different
    chains collapsed onto ONE entry, inverting the per-stage ranking CACHE-7 reads in v0.32."""
    print("\n--- v0.31 PROF-1: two different fused chains are two keys ---")
    with _armed():
        A = make_img(1, 96, 96, 4, seed=31)
        term = "@OUT = vec4(@IN.rgb + vec3(0.01), 1.0);"
        for tail in ("@OUT = gauss_blur(@IN, 3.0);", "@OUT = vec4(@IN.rgb * 0.5, 1.0);"):
            spec = {"schema": 1,
                    "stages": [{"code": "@OUT = vec4(@IN.rgb * 1.15, 1.0);",
                                "image_input": "IN", "params": {}},
                               {"code": tail, "image_input": "IN", "params": {}}],
                    "terminal_image_input": "IN"}
            tex_engine.cook(term, {"IN": A}, chain_payload=spec, device_mode="cpu")
        keys = list(P.snapshot())
        degenerate = [k for k in keys if k.startswith("None|")]
        r.ok(f"two chains -> {len(keys)} keys") if len(keys) == 2 else \
            r.fail("PROF-1 fused key", f"{len(keys)} key(s): {keys}")
        r.ok("no key degenerates to None|device|precision") if not degenerate else \
            r.fail("PROF-1 fused key", f"degenerate keys: {degenerate}")


def test_v031_prof1_state_is_thread_safe(r: SubTestResult):
    """`_STATE` has two real mutators in the SHIPPED config — `tex_engine.run` on the cook thread
    when armed, and the cook queue's worker feeding job timings even while DISARMED — and no
    lock. `snapshot()` iterating while the worker inserts raised `RuntimeError: OrderedDict
    mutated during iteration` (41-68 hits per 4 s). `enabled()` stays outside the lock so the
    default cook path is untouched."""
    print("\n--- v0.31 PROF-1: concurrent readers and writers ---")
    P.reset()
    try:
        stop, errs = threading.Event(), []

        def writer(n):
            i = 0
            while not stop.is_set():
                try:
                    P.record(P.make_key(f"race{n}-{i % 97}", "cpu", "fp32"), 1.0 + i % 5, 4096)
                except BaseException as e:              # noqa: BLE001
                    errs.append(repr(e))
                i += 1

        def reader():
            while not stop.is_set():
                try:
                    P.snapshot()
                    P.predict(P.make_key("race0-1", "cpu", "fp32"), 4096)
                    P.stage_costs(P.make_key("race0-1", "cpu", "fp32"), 4096)
                except BaseException as e:              # noqa: BLE001
                    errs.append(repr(e))

        threads = [threading.Thread(target=writer, args=(i,), daemon=True) for i in range(3)] + \
                  [threading.Thread(target=reader, daemon=True) for _ in range(2)]
        for t in threads:
            t.start()
        time.sleep(1.5)
        stop.set()
        for t in threads:
            t.join(3.0)
        r.ok("3 writers + 2 readers for 1.5 s: no errors") if not errs else \
            r.fail("PROF-1 thread safety", f"{len(errs)} error(s), first: {errs[0]}")
    finally:
        P.reset()


def test_v031_pred1_unknown_cost_has_a_confidence_brake(r: SubTestResult):
    print("\n--- v0.31 PRED-1: an unmeasured program faces the confidence brake ---")
    policy = _pol(unknown_cost_ms=8.0, unknown_min_confidence=0.5, min_value_ms=2.0,
                  predict=lambda k, px: None)          # PROF-1 knows nothing
    low = Q.Job(id=1, klass=Q.SPECULATIVE, fn=lambda c: None, confidence=0.4,
                profile_key=("x", "cpu", "fp32"), reason=Q.PLAY_HOVER)
    high = Q.Job(id=2, klass=Q.SPECULATIVE, fn=lambda c: None, confidence=0.8,
                 profile_key=("x", "cpu", "fp32"), reason=Q.PANEL_OPEN)
    r.ok("unmeasured + low confidence -> refused") if not policy.admit(low) else \
        r.fail("PRED-1 unknown", "an unmeasured low-confidence job was admitted")
    r.ok("unmeasured + high confidence -> admitted (a cold session must be able to learn)") \
        if policy.admit(high) else \
        r.fail("PRED-1 unknown", "an unmeasured high-confidence job was refused")
    got = policy.refusals.get(Q.PLAY_HOVER)
    r.ok(f"a refusal is counted AND explained, by reason: {policy.refusals}") \
        if got and got[0] == 1 and "confidence brake" in got[1] else \
        r.fail("PRED-1 refusals", f"{policy.refusals}")


def test_v031_pred1_never_touches_the_other_classes(r: SubTestResult):
    """CANARY: the policy scores SPECULATIVE and nothing else. An INTERACTIVE frame at
    confidence 0 and cost 0 is still cooked — the user asked for it."""
    print("\n--- v0.31 PRED-1: INTERACTIVE and COMMITTED are never scored ---")
    policy = _pol(min_value_ms=1e9)                    # a floor nothing could clear
    for klass, name in ((Q.INTERACTIVE, "INTERACTIVE"), (Q.COMMITTED, "COMMITTED")):
        job = Q.Job(id=1, klass=klass, fn=lambda c: None, confidence=0.0, cost_ms=0.0)
        r.ok(f"{name} is admitted under an impossible floor") if policy.admit(job) else \
            r.fail("PRED-1 exemption", f"{name} was refused by the speculative policy")

    with Q.CookQueue() as live:
        live.install_policy(_pol(min_value_ms=1e9))
        got = live.submit(lambda c: "frame", klass=Q.INTERACTIVE).result(timeout=_WAIT)
        r.ok("an interactive submit still runs with a hostile policy installed") \
            if got == "frame" else r.fail("PRED-1 exemption", f"got {got!r}")
        spec = live.submit(lambda c: "spec", klass=Q.SPECULATIVE, confidence=1.0, cost_ms=1.0)
        r.ok("…and the speculative submit is refused, with a counter") \
            if spec.state == Q.CANCELLED and live.snapshot()["stats"]["refused"] == 1 else \
            r.fail("PRED-1 refusal", f"state={spec.state} stats={live.snapshot()['stats']}")


def test_v031_pred1_orders_and_sheds_by_value(r: SubTestResult):
    print("\n--- v0.31 PRED-1: highest value first, lowest value shed first ---")
    gate, order = threading.Event(), []
    with Q.CookQueue() as q:
        q.install_policy(_pol(min_value_ms=0.5, max_pending=99))
        q.submit(lambda c: gate.wait(_WAIT), klass=Q.INTERACTIVE)   # hold the worker
        time.sleep(0.02)
        # Submitted worst-first; they must RUN best-first.
        for conf, cost, tag in ((0.2, 10.0, "low"), (0.9, 50.0, "high"), (0.5, 20.0, "mid")):
            q.submit((lambda t: lambda c: order.append(t))(tag), klass=Q.SPECULATIVE,
                     confidence=conf, cost_ms=cost, reason=Q.NEIGHBOR_FRAME)
        gate.set()
        q.drain(timeout=_WAIT)
        r.ok(f"speculative work runs highest-value first: {order}") \
            if order == ["high", "mid", "low"] else \
            r.fail("PRED-1 order", f"{order}")

    # Shedding: overflow max_pending and the lowest scores go, the best survive.
    gate = threading.Event()
    with Q.CookQueue() as q:
        q.install_policy(_pol(min_value_ms=0.1, max_pending=3))
        q.submit(lambda c: gate.wait(_WAIT), klass=Q.INTERACTIVE)
        time.sleep(0.02)
        jobs = [q.submit(lambda c: None, klass=Q.SPECULATIVE, confidence=1.0,
                         cost_ms=float(i + 1), reason=Q.PREFETCH) for i in range(8)]
        alive = [j for j in jobs if j.state != Q.CANCELLED]
        shed = [j for j in jobs if j.state == Q.CANCELLED]
        gate.set()
        q.drain(timeout=_WAIT)
        r.ok(f"max_pending=3 keeps 3 of 8 ({[j.cost_ms for j in alive]})") \
            if len(alive) == 3 else \
            r.fail("PRED-1 shed", f"{len(alive)} survived, expected 3")
        r.ok("the survivors are the highest-scoring ones") \
            if alive and min(j.cost_ms for j in alive) > max(j.cost_ms for j in shed) else \
            r.fail("PRED-1 shed order",
                   f"kept {[j.cost_ms for j in alive]}, shed {[j.cost_ms for j in shed]}")
        r.ok("shed work is terminal CANCELLED and counted") \
            if q.snapshot()["stats"]["shed"] == len(shed) else \
            r.fail("PRED-1 shed count", f"{q.snapshot()['stats']}")


def test_v031_pred1_sheds_the_worst_even_after_a_requeue(r: SubTestResult):
    """The shed must find the LOWEST-scoring job by value, not by position.

    An earlier version popped the deque's tail, reasoning that `_enqueue_locked` keeps it in
    descending score order. Head-requeue breaks that: a preempted job goes to the FRONT
    regardless of score, so once the tail grows past it the tail is the MAXIMUM — and the shed
    dropped the single most valuable bet while keeping the worst. Reachable from the shipped
    policy alone, since its `shed()` runs on every submit.

    So: get a low-scoring job head-requeued, queue a high-scoring one behind it, and shed."""
    print("\n--- v0.31 PRED-1: the shed victim is the lowest SCORE, not the tail ---")
    started, release = threading.Event(), threading.Event()

    def blocker(cancel):
        started.set()
        while not release.wait(0.002):
            cancel.check()
        return "low"

    with Q.CookQueue(min_quantum_ms=0.0) as q:
        q.install_policy(_pol(min_value_ms=0.0, min_confidence=0.0, max_pending=10 ** 6))
        low = q.submit(blocker, klass=Q.SPECULATIVE, cost_ms=1.0, reason=Q.PREFETCH)
        if not started.wait(_WAIT):
            r.fail("PRED-1 shed order", "the low job never started")
            return
        # Preempt it, so it is head-requeued at the FRONT with the lowest score in the queue.
        hold = threading.Event()
        q.submit(lambda c: hold.wait(_WAIT), klass=Q.INTERACTIVE)
        for _ in range(int(_WAIT / 0.002)):
            if q.snapshot()["stats"]["requeued"] >= 1:
                break
            time.sleep(0.002)
        high = q.submit(lambda c: "high", klass=Q.SPECULATIVE, cost_ms=100.0,
                        reason=Q.NEIGHBOR_FRAME)
        r.ok(f"a requeued low-score job sits at the head (requeued="
             f"{q.snapshot()['stats']['requeued']})") \
            if q.snapshot()["stats"]["requeued"] >= 1 else \
            r.fail("PRED-1 shed order", "the preempt never landed")

        q.shed_speculative(keep=1)
        r.ok("the 1 ms bet is shed and the 100 ms bet survives") \
            if low.state == Q.CANCELLED and high.state != Q.CANCELLED else \
            r.fail("PRED-1 shed order",
                   f"shed the wrong job: low={low.state} high={high.state}")
        release.set()
        hold.set()
        q.drain(timeout=_WAIT)


def test_v031_pred1_closes_the_loop_with_prof1(r: SubTestResult):
    """The queue brackets every job anyway, so a submitter who names a profile key gets the
    measurement fed back — no engine involvement, nothing on the default cook path. Without
    this, PRED-1's cost oracle would only ever learn from cooks someone else profiled."""
    print("\n--- v0.31 PRED-1 <-> PROF-1: a completed job feeds the cost oracle ---")
    P.reset()
    try:
        key = P.make_key("loop-probe", "cpu", "fp32")
        r.ok("the oracle starts empty for this key") if P.predict(key, 4096) is None else \
            r.fail("PRED-1 loop", "the key was already known")
        with Q.CookQueue() as q:
            q.submit(lambda c: time.sleep(0.01), klass=Q.SPECULATIVE,
                     profile_key=key, px=4096, confidence=1.0,
                     reason=Q.IDLE_CHECKPOINT).result(timeout=_WAIT)
        ms = P.predict(key, 4096)
        r.ok(f"after one completed job the oracle predicts {ms:.1f} ms") \
            if ms is not None and ms >= 9.0 else \
            r.fail("PRED-1 loop", f"predict returned {ms!r} after a ~10 ms job")

        # And it is fed even though the profiler is DISARMED — the queue already paid for the
        # timing, so throwing it away because the in-engine sampler is off would be silly.
        r.ok("the loop works with the in-engine profiler disarmed") if not P.enabled() else \
            r.fail("PRED-1 loop", "this test armed the profiler and did not disarm it")
    finally:
        P.reset()
