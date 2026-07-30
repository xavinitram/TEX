"""v0.34.1 — the v0.34.0 release-audit findings (doc 42 §1), each pinned by a row verified
to fail on a pristine `git worktree` at v0.34.0 (9/9 reproduced pre-fix, 0/9 after).

Same shape and standard as v0.33.1/v0.33.2: every one of these lives behind a
REGISTERED-PROVIDER or QUEUE-WITH-PROMISES path, so none is reachable from a default ComfyUI
cook — and every one is closed anyway, because "unreachable today" is a property of the
wiring, not of the code.

  A  `set_provider` racing an in-flight fetch served the REPLACED provider's pixels under the
     new provider's key — silently, because the default `provider_id` is the class name and
     re-registering the same decoder class makes the two ids equal
  B  `cancel()` of a WAITING job never terminated it: `wait()`/`drain()` hung, and a later
     landing flipped the cancelled job to PENDING and GRANTED ITS DEFERRED PREEMPTION,
     destroying a running COMMITTED render's progress for a job that can never run
  C  `_normalize` never checked dtype — an f64 provider frame reached the cook output under
     `precision="fp32"` and dragged whole expressions to f64
  D  all-scalar coordinates took the PROVIDER's device (a rank test where a device test was
     meant), so a CPU provider in a CUDA cook raised a bare RuntimeError with no E-code; and
     the const-coord result was [1,1,1,C] where every comparator produces the cook grid
  E  the media pool accounted a VIEW's numel, so a provider returning `clip[i:i+1]`
     under-reported 64x and pinned the whole clip past every eviction
  F  a promise that FAILED before its job ran alarmed even for SPECULATIVE work — the
     class-dependent rule lived only in `_run_one`, not in the wake path
  G  `Promise.land(None)` half-landed: success returned, `landed` still False, callbacks
     consumed — a later correct `land()` raised and woke nobody. Permanent park.
  H  the stage-list family neither resolved nor refused Promises (raw TypeError), and
     `boundary_lineage_key` folded one through `repr` — ADDRESS-KEYED checkpoint identity
  I  a >=5-D tensor still typed FLOAT inside the tensor branch, the one residue of the
     pre-E7005 guess
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


def test_v0341_a_a_provider_swap_never_serves_the_replaced_pixels(r):
    """A fetch that spans `set_provider` must not be indexed under the new provider.

    Dropping the pool at swap time is not enough on its own: `materialize` captured the old
    provider before the swap and `put()`s after it. A generation counter stamped at entry is
    what covers the window the drop cannot reach."""
    class Slow:
        provider_id = "shared"          # the ordinary case: same decoder class, same id

        def __init__(self, v):
            self.v = v

        def fetch_time(self, k, t):
            time.sleep(0.2)
            return torch.full((1, 8, 8, 4), self.v)

        sample_time = fetch_time

    tex_provider.reset_provider()
    tex_provider.set_provider(Slow(1.0))
    try:
        th = threading.Thread(target=lambda: tex_provider.materialize("p", 0.0, "fetch"))
        th.start()
        time.sleep(0.05)
        tex_provider.set_provider(Slow(2.0))
        th.join(10)
        served = float(tex_provider.materialize("p", 0.0, "fetch").flatten()[0])
        if served == 2.0:
            r.ok("A: a fetch spanning a provider swap is never indexed under the new provider")
        else:
            r.fail("v0.34.1 A provider swap",
                   f"the pool served {served} after swapping 1.0 -> 2.0")
    except Exception as e:
        r.fail("v0.34.1 A provider swap", f"{type(e).__name__}: {e}")
    finally:
        tex_provider.reset_provider()


def test_v0341_b_cancelling_a_waiting_job_terminates_it(r):
    """`cancel()` on WAITING is terminal, and a later landing does not resurrect it.

    Two defects in one: the hang (wait()/drain() never returned) and the far worse one — the
    landing granted the preemption the waiting submit had been denied, so a running COMMITTED
    render lost ALL its progress (there is no resume) for a job the host had abandoned."""
    q = Q.CookQueue(name="v0341-b", min_quantum_ms=0.0)
    try:
        p = Promise("A", type=TEXType.VEC4)
        job = q.submit(lambda c: "never", klass=Q.COMMITTED, inputs=[p])
        assert q.cancel(job) is True
        assert job.wait(5), "cancel() of a WAITING job did not terminate it"
        assert job.state == Q.CANCELLED, job.state
        assert q.stats.waiting == 0, q.stats.waiting
        assert q.drain(5), "drain() hung on a cancelled waiting job"

        p.land(_img())                    # the resurrection attempt
        time.sleep(0.15)
        assert job.state == Q.CANCELLED, f"a landing resurrected a cancelled job: {job.state}"
        assert q.stats.preempted == 0, \
            f"a cancelled job's landing granted {q.stats.preempted} preemption(s)"
        r.ok("B: cancelling a WAITING job is terminal; a later landing cannot resurrect it")
    except Exception as e:
        r.fail("v0.34.1 B cancel WAITING", f"{type(e).__name__}: {e}")
    finally:
        q.close()


def test_v0341_c_a_provider_frame_is_fp32_or_refused(r):
    """Float dtypes convert at the pool boundary; integer dtypes are refused.

    The split is by KIND. Half is what an EXR source legitimately decodes to and fp32 is the
    precision the cook asked for, so floats convert. Turning uint8 into float means choosing
    a normalization (/255? sRGB-decoded?) — a colour decision this seam must never make on
    the host's behalf."""
    class Typed:
        provider_id = "typed"

        def __init__(self, dt):
            self.dt = dt

        def fetch_time(self, k, t):
            if self.dt.is_floating_point:
                return torch.full((1, 8, 8, 4), 0.5, dtype=self.dt)
            return torch.full((1, 8, 8, 4), 128, dtype=self.dt)

        sample_time = fetch_time

    try:
        for dt in (torch.float64, torch.float16):
            tex_provider.reset_provider()
            tex_provider.set_provider(Typed(dt))
            a = tex_engine.cook('@OUT = fetch_time("p", 0.0, ix, iy);', {"A": _img()},
                                device_mode="cpu").outputs["OUT"]
            b = tex_engine.cook('@OUT = sample_time("p", 0.0, u, v);', {"A": _img()},
                                device_mode="cpu").outputs["OUT"]
            assert a.dtype == torch.float32, f"{dt}: fetch_time -> {a.dtype}"
            assert b.dtype == torch.float32, f"{dt}: sample_time -> {b.dtype}"

        tex_provider.reset_provider()
        tex_provider.set_provider(Typed(torch.uint8))
        try:
            tex_engine.cook('@OUT = fetch_time("p", 0.0, ix, iy);', {"A": _img()},
                            device_mode="cpu")
            r.fail("v0.34.1 C dtype", "an integer provider frame was accepted")
            return
        except Exception as e:
            assert getattr(e, "_code", "") == tex_provider.E_BAD_FRAME, str(e)
        r.ok("C: float provider frames convert to fp32; integer frames are refused as E7004")
    except Exception as e:
        r.fail("v0.34.1 C dtype", f"{type(e).__name__}: {e}")
    finally:
        tex_provider.reset_provider()


def test_v0341_d_constant_coordinates_stay_on_the_cook_grid(r):
    """Const coords produce the cook grid on the cook's device, like every comparator.

    The device half was a RANK test standing in for a device test: the interpreter evaluates
    a numeric literal into a 0-dim tensor ON THE COOK DEVICE, and testing `.dim()` threw that
    away. The shape half is the alignment doc 42 asks for — `vec4(...)` and `fetch(@A,4,4)`
    both yield the cook grid, and `fetch_time` yielded [1,1,1,C]."""
    for dev in _devices():
        tex_provider.reset_provider()
        tex_provider.set_provider(tex_provider.SyntheticFrameProvider(res=8, device="cpu"))
        try:
            out = tex_engine.cook('@OUT = fetch_time("p", 0.0, 4, 4);',
                                  {"A": _img(16, dev=dev)}, device_mode=dev).outputs["OUT"]
            assert tuple(out.shape) == (1, 16, 16, 4), tuple(out.shape)
            assert out.device.type == torch.device(dev).type, out.device
            # ...and it composes with cook-grid values rather than raising a bare RuntimeError
            mixed = tex_engine.cook(
                '@OUT = fetch_time("p", 0.0, 4, 4) * 0.5 + vec4(@A.rgb, 1.0) * 0.5;',
                {"A": _img(16, dev=dev)}, device_mode=dev).outputs["OUT"]
            assert tuple(mixed.shape) == (1, 16, 16, 4), tuple(mixed.shape)
            r.ok(f"D: constant coordinates yield the cook grid on the cook device ({dev})")
        except Exception as e:
            r.fail(f"v0.34.1 D const coords ({dev})", f"{type(e).__name__}: {e}")
        finally:
            tex_provider.reset_provider()


def test_v0341_e_the_pool_owns_the_bytes_it_accounts(r):
    """A pooled frame owns exactly its storage — it is never a view of a host buffer.

    A provider returning `clip[i:i+1]` pinned the whole clip while accounting one frame, so
    the governor priced the pool 64x wrong AND `evict_bytes` "freed" bytes that stayed
    resident. The copy also stops a host that keeps mutating its decode buffer from rewriting
    frames already in the pool."""
    clip = torch.rand(64, 8, 8, 4)

    class Viewer:
        provider_id = "viewer"

        def fetch_time(self, k, t):
            return clip[int(t):int(t) + 1]

        sample_time = fetch_time

    tex_provider.reset_provider()
    tex_provider.set_provider(Viewer())
    try:
        tex_provider.materialize("clip", 0.0, "fetch")
        cache = tex_provider.get_media_cache()
        entry = next(iter(cache._entries.values()))
        held = entry.tensor.untyped_storage().nbytes()
        assert cache.stats()["bytes"] == held, \
            f"accounted {cache.stats()['bytes']} B, holds {held} B"
        assert entry.tensor.untyped_storage().data_ptr() != clip.untyped_storage().data_ptr(), \
            "the pooled frame still aliases the host's clip"
        before = float(entry.tensor.flatten()[0])
        clip.zero_()
        assert float(entry.tensor.flatten()[0]) == before, \
            "a host mutation reached a frame already in the pool"
        r.ok("E: a pooled frame owns exactly the bytes the governor is told about")
    except Exception as e:
        r.fail("v0.34.1 E pool accounting", f"{type(e).__name__}: {e}")
    finally:
        tex_provider.reset_provider()


def test_v0341_f_a_failed_promise_never_alarms_speculative_work(r):
    """The class-dependent host-I/O rule applies at the WAKE path too, not only in `_run_one`.

    A prefetch whose source failed before the job ever ran reported FAILED and handed a raw
    E7xxx to the waiter, with no refusals-ledger row — the opposite of "speculation never
    alarms", through the one door the original fix did not cover."""
    from TEX_Wrangle.tex_runtime.interpreter import InterpreterError
    q = Q.CookQueue(name="v0341-f")
    q.install_policy(Q.SpeculativePolicy(min_confidence=0.0, min_value_ms=0.0,
                                         unknown_min_confidence=0.0))
    try:
        p = Promise("A", type=TEXType.VEC4)
        job = q.submit(lambda c: None, klass=Q.SPECULATIVE, reason=Q.PREFETCH,
                       confidence=0.9, inputs=[p])
        p.fail(InterpreterError("the plate is gone", None, code="E7002"))
        assert job.wait(5), "the job never finished"
        assert job.state == Q.CANCELLED, job.state
        assert q.stats.failed == 0, f"a speculative I/O failure was counted as failed"
        led = q._policy.refusals.get(Q.PREFETCH)
        assert led and "host I/O failed" in led[1], q._policy.refusals

        # The control: a COMMITTED job with the same failure still reports it.
        p2 = Promise("A", type=TEXType.VEC4)
        boom = InterpreterError("gone", None, code="E7002")
        j2 = q.submit(lambda c: None, klass=Q.COMMITTED, inputs=[p2])
        p2.fail(boom)
        j2.wait(5)
        assert j2.state == Q.FAILED and j2.error is boom, (j2.state, j2.error)
        r.ok("F: a failed promise ledgers for SPECULATIVE work and raises for COMMITTED")
    except Exception as e:
        r.fail("v0.34.1 F speculative wake failure", f"{type(e).__name__}: {e}")
    finally:
        q.close()


def test_v0341_g_a_promise_cannot_land_none(r):
    """`land(None)` is refused, not half-accepted.

    `landed` is derived from `_value is not None`, so landing None returned success, left
    `landed` False, and consumed the callback list — after which the correct `land()` raised
    "already landed" and woke nobody. A permanent park from a call that looked fine."""
    try:
        p = Promise("A", type=TEXType.FLOAT)
        seen = []
        p.on_land(lambda _p: seen.append(1))
        try:
            p.land(None)
            r.fail("v0.34.1 G land(None)", "None was accepted")
            return
        except Exception as e:
            # E7006, not a bare ValueError: a promise-declaration violation like every other
            # one `_check` raises, so a host routing E7xxx still catches it.
            assert getattr(e, "_code", "") == "E7006", f"{getattr(e, '_code', '')}: {e}"
            assert "fail(exc)" in str(e) or "fail(exc)" in getattr(e, "_hint", ""), str(e)
        assert not p.landed and not seen, (p.landed, seen)
        # ...and the promise is still usable afterwards, which is the whole point.
        p.land(1.5)
        assert p.landed and seen == [1], (p.landed, seen)
        r.ok("G: land(None) is refused and leaves the promise usable")
    except Exception as e:
        r.fail("v0.34.1 G land(None)", f"{type(e).__name__}: {e}")


def test_v0341_h_the_stage_list_family_understands_promises(r):
    """`cook_stage_list` resolves a landed promise, refuses an unlanded one, and
    `boundary_lineage_key` keys two equivalent promises IDENTICALLY.

    The identity half is the dangerous one: a Promise fell to the params side of the
    tensor/param split and `_canon_params` folds unknown objects through `repr`, which for a
    `__slots__` object is its ADDRESS — spurious checkpoint misses, and aliasing on address
    reuse."""
    try:
        src = _img()
        p = Promise("A", type=TEXType.VEC4, shape=tuple(src.shape))
        p.land(src)
        got = tex_engine.cook_stage_list(
            [{"code": "@OUT = vec4(@A.rgb, 1.0);", "bindings": {"A": p}}], device="cpu")["OUT"]
        direct = tex_engine.cook_stage_list(
            [{"code": "@OUT = vec4(@A.rgb, 1.0);", "bindings": {"A": src}}], device="cpu")["OUT"]
        assert torch.equal(got, direct), "cooking a stage list through a promise moved pixels"

        unlanded = Promise("A", type=TEXType.VEC4)
        try:
            tex_engine.cook_stage_list(
                [{"code": "@OUT = vec4(@A.rgb, 1.0);", "bindings": {"A": unlanded}}],
                device="cpu")
            r.fail("v0.34.1 H stage list", "an unlanded promise cooked")
            return
        except Exception as e:
            assert getattr(e, "_code", "") == "E7007", f"{getattr(e, '_code', '')}: {e}"

        def _key(promise):
            return tex_engine.boundary_lineage_key(
                [{"code": "@OUT = vec4(@A.rgb,1.0);", "bindings": {"A": promise}},
                 {"code": "@OUT = @A * 2.0;", "bindings": {}}],
                1, "cpu", "fp32", upstream=("u",))

        q = Promise("A", type=TEXType.VEC4, shape=tuple(src.shape))
        q.land(src.clone())
        assert _key(p) == _key(q), "two equivalent promises minted different checkpoint keys"
        r.ok("H: the stage-list family resolves/refuses promises and keys them by value")
    except Exception as e:
        r.fail("v0.34.1 H stage list promises", f"{type(e).__name__}: {e}")


def test_v0341_i_high_rank_tensors_and_helper_locations(r):
    """A >=5-D tensor is E7005, and a helper-raised error adopts its call site.

    The rank case was the last silent `return TEXType.FLOAT` — inside the tensor branch,
    where the E7005 terminal never sees it. The location case is why a fused E7002 could not
    name the stage that failed: `loc.stage` is the Q-4 marker, and the provider seam raises
    from below the AST where it cannot know one."""
    try:
        try:
            t = infer_binding_type(torch.rand(1, 2, 3, 4, 5))
            r.fail("v0.34.1 I rank", f"a 5-D tensor typed {t}")
            return
        except Exception as e:
            assert getattr(e, "_code", "") == "E7005", f"{getattr(e, '_code', '')}: {e}"
        # The ranks that must KEEP their meaning.
        assert infer_binding_type(torch.rand(1, 4, 4, 3)) is TEXType.VEC3
        assert infer_binding_type(torch.rand(1, 4, 4)) is TEXType.FLOAT
        assert infer_binding_type(torch.tensor(1.5)) is TEXType.FLOAT

        class Broken:
            provider_id = "broken"

            def fetch_time(self, k, t):
                raise OSError("gone")

            sample_time = fetch_time

        tex_provider.reset_provider()
        tex_provider.set_provider(Broken())
        try:
            tex_engine.cook('@OUT = vec4(0.0,0.0,0.0,1.0);\n'
                            '@OUT = fetch_time("p", 0.0, ix, iy);',
                            {"A": _img()}, device_mode="cpu")
            r.fail("v0.34.1 I loc", "a broken provider did not raise")
            return
        except Exception as e:
            assert getattr(e, "_code", "") == "E7002", getattr(e, "_code", "")
            assert e.loc is not None and e.loc.line == 2, \
                f"E7002 carried loc={e.loc} (want the calling line, 2)"
        r.ok("I: >=5-D tensors are E7005; a helper's error adopts its call site")
    except Exception as e:
        r.fail("v0.34.1 I closures", f"{type(e).__name__}: {e}")
    finally:
        tex_provider.reset_provider()


# ── the re-audit's holes: each fix's shipped row pinned the interleaving the fix chose ──
#
# Every row below covers the case that SLIPPED, not the case that was reported. They are kept
# separate from the rows above on purpose: those pin the reported defect (and still should),
# these pin the hole the first draft left, and merging them is how a hole inherits a green row.

def test_v0341_h_promise_prefixes_key_by_resolution(r):
    """P0-H, the REGRESSION half. Promise-fed prefixes must key by resolution, and an
    uncovered promise must never reach the cache.

    Removing Promise from the params fold without adding it to the tensor side made a
    promise-fed prefix invisible to BOTH halves of `boundary_lineage_key`: `_shapes()` skipped
    it, the canvas came back empty, fell through to `chain_in`, came back empty again — one
    key for every resolution. Address-keying (v0.34.0) was wasteful but SAFE; one key for two
    resolutions is a wrong-size boundary served on a cache HIT, which is exactly what the
    P0-3 comment beside it says was already closed."""
    from TEX_Wrangle import tex_results

    def _key(res):
        p = Promise("A", type=TEXType.VEC4, shape=(1, res, res, 4))
        p.land(_img(res))
        return tex_engine.boundary_lineage_key(
            [{"code": "@OUT = vec4(@A.rgb,1.0);", "bindings": {"A": p}},
             {"code": "@OUT = @A * 2.0;", "bindings": {}}], 1, "cpu", "fp32", upstream=("u",))

    try:
        assert _key(8) != _key(16), "an 8x8 and a 16x16 promise prefix minted the SAME key"

        a = Promise("A", type=TEXType.VEC4, shape=(1, 8, 8, 4))
        a.land(_img())
        b = Promise("A", type=TEXType.VEC4, shape=(1, 8, 8, 4))
        b.land(_img())

        def _k(p):
            return tex_engine.boundary_lineage_key(
                [{"code": "@OUT = vec4(@A.rgb,1.0);", "bindings": {"A": p}},
                 {"code": "@OUT = @A * 2.0;", "bindings": {}}], 1, "cpu", "fp32",
                upstream=("u",))
        assert _k(a) == _k(b), "two equivalent promises still key differently"

        # The gate: upstream=() covers zero tensors while the prefix reads one (promised).
        p = Promise("A", type=TEXType.VEC4, shape=(1, 8, 8, 4))
        p.land(_img())
        cache = tex_results.ResultCache(budget_mb=64)
        tex_engine.cook_fused_cached(
            [{"code": "@OUT = vec4(@A.rgb,1.0);", "bindings": {"A": p}},
             {"code": "@OUT = @A * 2.0;", "bindings": {}, "chain_inputs": {"A": [0, "OUT"]}}],
            1, cache, device="cpu", upstream=())
        assert len(cache._ram) == 0, \
            f"the coverage gate admitted an UNCOVERED promise prefix ({len(cache._ram)} cached)"

        # An unlanded, shapeless promise is un-keyable — refuse, never guess a resolution.
        blind = Promise("A", type=TEXType.VEC4)
        cache2 = tex_results.ResultCache(budget_mb=64)
        try:
            tex_engine.cook_fused_cached(
                [{"code": "@OUT = vec4(@A.rgb,1.0);", "bindings": {"A": blind}},
                 {"code": "@OUT = @A*2.0;", "bindings": {}, "chain_inputs": {"A": [0, "OUT"]}}],
                1, cache2, device="cpu", upstream=("u",))
        except Exception as e:
            assert getattr(e, "_code", "") == "E7007", f"{getattr(e, '_code', '')}: {e}"
        assert len(cache2._ram) == 0, "a shapeless promise prefix was cached anyway"

        # The line above passes through `_full()`'s E7007 whether or not the GATE refuses, so
        # it does not actually pin the shapeless term. The refusal that matters is in
        # `boundary_lineage_key` itself, where a caller reaching it directly would otherwise
        # get a key minted over a resolution nobody knows.
        try:
            tex_engine.boundary_lineage_key(
                [{"code": "@OUT = vec4(@A.rgb,1.0);", "bindings": {"A": blind}},
                 {"code": "@OUT = @A * 2.0;", "bindings": {}}], 1, "cpu", "fp32",
                upstream=("u",))
            r.fail("v0.34.1 H-hole", "a shapeless promise was keyed instead of refused")
            return
        except ValueError as e:
            assert "declared no shape" in str(e), str(e)
        r.ok("H-hole: promise prefixes key by resolution; shapeless and uncovered both refuse")
    except Exception as e:
        r.fail("v0.34.1 H-hole", f"{type(e).__name__}: {e}")


def test_v0341_d_the_cook_grid_agrees_across_tiers(r):
    """P0-D, the divergence half. Const-coord reads give the same shape on every tier.

    Publishing the grid in `Interpreter.execute` alone did not half-fix the defect, it created
    a fresh one: the same program returned [1,16,16,4] under `compile_mode="none"` and
    [1,1,1,4] under `"auto"` — an interp/codegen split (invariant #2) where v0.34.0 at least
    had both tiers agreeing on being wrong. The shipped row passed because it only ever
    exercised the default tier."""
    for dev in _devices():
        tex_provider.reset_provider()
        tex_provider.set_provider(tex_provider.SyntheticFrameProvider(res=8, device="cpu"))
        try:
            code = '@OUT = fetch_time("p", 0.0, 4, 4);'
            shapes = {}
            for mode in ("none", "auto"):
                out = tex_engine.cook(code, {"A": _img(16, dev=dev)}, device_mode=dev,
                                      compile_mode=mode).outputs["OUT"]
                shapes[mode] = tuple(out.shape)
            assert shapes["none"] == shapes["auto"] == (1, 16, 16, 4), shapes
            r.ok(f"D-hole: the const-coord grid agrees across interp and codegen ({dev})")
        except Exception as e:
            r.fail(f"v0.34.1 D-hole ({dev})", f"{type(e).__name__}: {e}")
        finally:
            tex_provider.reset_provider()


def test_v0341_e_a_buffer_reusing_provider_is_still_copied(r):
    """P0-E, the mainstream half. The fast path exempted exactly the ordinary provider.

    `owns_exactly` skipped the copy for a contiguous tensor owning its storage — which is
    precisely what a provider decoding into a reusable buffer hands back, so both symptoms the
    fix claimed to close survived it. Ownership is a promise about the FUTURE, not a property
    readable off a tensor: the default is to copy, and a host opts out by declaring it."""
    buf = torch.zeros(1, 8, 8, 4)

    class Reuser:
        provider_id = "reuser"

        def fetch_time(self, k, t):
            buf.fill_(float(t) + 1.0)
            return buf

        sample_time = fetch_time

    tex_provider.reset_provider()
    tex_provider.set_provider(Reuser())
    try:
        f0 = tex_provider.materialize("s", 0.0, "fetch")
        v0 = float(f0.flatten()[0])
        tex_provider.materialize("s", 1.0, "fetch")       # rewrites the shared buffer
        assert float(f0.flatten()[0]) == v0, \
            "a buffer-reusing provider rewrote a frame already in the pool"
        st = tex_provider.get_media_cache().stats()
        assert st["copies"] >= 2, st
        copies_before = st["copies"]

        class Owned:
            provider_id = "owned"
            frames_are_owned = True

            def fetch_time(self, k, t):
                return torch.full((1, 8, 8, 4), float(t) + 1.0)

            sample_time = fetch_time

        tex_provider.reset_provider()
        tex_provider.set_provider(Owned())
        tex_provider.materialize("s", 0.0, "fetch")
        # DELTA, not absolute: `reset_provider()` clears the pool's entries but deliberately
        # not its lifetime counters, so an absolute compare here was reading the Reuser
        # phase's copies and calling them the Owned phase's.
        assert tex_provider.get_media_cache().stats()["copies"] == copies_before, \
            "frames_are_owned did not suppress the defensive copy"
        r.ok("E-hole: a buffer-reusing provider is copied; a declared-owned one is not")
    except Exception as e:
        r.fail("v0.34.1 E-hole", f"{type(e).__name__}: {e}")
    finally:
        tex_provider.reset_provider()


def test_v0341_c_a_half_source_stays_half_in_the_pool(r):
    """P0-C, the fp16 half. Forcing fp32 at the pool boundary un-did `precision="fp16"`.

    `_normalize` cannot see the cook's precision, so hardcoding fp32 there DOUBLED the pool
    for a half-float EXR source — the exact case the fix's own comment names as legitimate.
    The pool now keeps the source's width and the cast happens at the read, where the cook's
    dtype is known."""
    class Half:
        provider_id = "half"

        def fetch_time(self, k, t):
            return torch.full((1, 64, 64, 4), 0.5, dtype=torch.float16)

        sample_time = fetch_time

    tex_provider.reset_provider()
    tex_provider.set_provider(Half())
    try:
        tex_provider.materialize("s", 0.0, "fetch")
        held = tex_provider.get_media_cache().stats()["bytes"]
        assert held == 64 * 64 * 4 * 2, f"pool holds {held} B for a half frame (want 32768)"
        out = tex_engine.cook('@OUT = fetch_time("s", 0.0, ix, iy);', {"A": _img(16)},
                              device_mode="cpu", precision="fp32").outputs["OUT"]
        assert out.dtype == torch.float32, out.dtype

        # ...and f64 IS still narrowed at the boundary. The read-side cast alone would give
        # the right output dtype, so the only thing the narrowing still buys is pool WIDTH —
        # which is exactly what a row asserting only the output dtype cannot see.
        class Wide:
            provider_id = "wide"

            def fetch_time(self, k, t):
                return torch.full((1, 64, 64, 4), 0.5, dtype=torch.float64)

            sample_time = fetch_time

        tex_provider.reset_provider()
        tex_provider.set_provider(Wide())
        tex_provider.materialize("s", 0.0, "fetch")
        wide = tex_provider.get_media_cache().stats()["bytes"]
        assert wide == 64 * 64 * 4 * 4, f"an f64 source pooled {wide} B (want 65536, narrowed)"
        r.ok("C-hole: a half source stays half, an f64 source is narrowed, both cast at the read")
    except Exception as e:
        r.fail("v0.34.1 C-hole", f"{type(e).__name__}: {e}")
    finally:
        tex_provider.reset_provider()


def test_v0341_g_fail_none_is_refused_too(r):
    """P0-G, the sibling. `fail(None)` has the identical half-land shape and is worse.

    `landed` derives from `_error is not None`, so `fail(None)` returned success, left the
    promise unlanded, and CONSUMED the callbacks — after which not even a later correct
    `land()` could wake the parked job. `drain()` never returns; `stats.waiting` sticks at 1.
    The first draft guarded `land()` and stopped there."""
    try:
        p = Promise("A", type=TEXType.VEC4)
        seen = []
        p.on_land(lambda _p: seen.append(1))
        try:
            p.fail(None)
            r.fail("v0.34.1 G-sibling", "fail(None) was accepted")
            return
        except Exception as e:
            assert getattr(e, "_code", "") == "E7006", f"{getattr(e, '_code', '')}: {e}"
        assert not p.landed and not seen, (p.landed, seen)
        p.land(_img())                       # still usable, callbacks intact
        assert p.landed and seen == [1], (p.landed, seen)

        q = Promise("B", type=TEXType.VEC4)
        try:
            q.land(None)
            r.fail("v0.34.1 G-sibling", "land(None) was accepted")
            return
        except Exception as e:
            assert getattr(e, "_code", "") == "E7006", f"{getattr(e, '_code', '')}: {e}"
        r.ok("G-hole: fail(None) and land(None) are both refused as routable E7006")
    except Exception as e:
        r.fail("v0.34.1 G-sibling", f"{type(e).__name__}: {e}")
