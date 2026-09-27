"""
COMPILE-A (v0.46) — a toolchain-aware, non-stalling "auto" tier.

AUTO-47 (v0.47) adds a fifth piece at the end, CC-7: the toolchain probe itself
(`compile_capability()`) still ran ON THE COOK THREAD the instant a key first became
eligible to compile — measured on an embedding host (a GPU box whose venv has no
Triton) at 1.4-1.8s on a quiet box, once 9.6s under GPU contention. `compile_capability_async()`
(`compiled_capability.py`) is the non-blocking counterpart `run_auto` now calls instead:
it returns the same dict once the probe (running on its own background thread) has
actually finished, else `None` immediately — a key that reads `None` just stays
MEASURING for a later cook to re-check, bounded the same way CC-6 already bounds any
other stuck key. `compile_capability()` itself is unchanged; every other caller
(a host diagnostic, `tex_api.compile_capability`) still gets a blocking, definitive
answer.

Four pieces, each landed as its own commit and covered here in the same order:

* **CC-3** — `compile_capability()`: a read-only, process-wide probe of whether
  torch.compile's inductor backend has its prerequisite (Triton on CUDA, a C compiler on
  CPU) — probed ONCE, never by compiling.
* **CC-4** — toolchain-aware "auto": when the target device's capability reads False,
  `run_auto` makes NO compile attempt at all (no failed-compile tax, no trial).
* **CC-5** — a compiled callable's LAZY first-call cost (Dynamo trace + Inductor/Triton
  lowering — `_try_compile` only wraps; nothing actually traces until the wrapped callable
  is first invoked) is paid on the BACKGROUND worker (`_submit_bg_compile`'s `warm_call`),
  never on the interactive cook thread's TRIAL tick.
* **CC-6** — bounded trial convergence: a key eligible to compile for more than
  `autotier._CONVERGENCE_BOUND_S` wall-clock seconds without reaching a terminal verdict
  is declined (REJECTED) rather than polled forever — a real measurement on a Triton-
  equipped box found a multi-stage program with several keys left "measuring" for well
  over a hundred seconds at high resolution, all contending for one process-wide
  background-compile worker.

PORTABILITY. CPU only, no CUDA required. CC-5's test simulates torch.compile's lazy
first-call cost with a monkeypatched fake compiled fn (`time.sleep`) rather than a real
compile — this box's Smart App Control policy forbids looping real `torch.compile` calls,
and the mechanism under test (which thread pays the lazy first-call cost) does not need a
real backend to prove.
"""
import time

import pytest

from helpers import *  # noqa: F401,F403
from TEX_Wrangle.tex_cache import parse_and_split
from TEX_Wrangle.tex_runtime import compiled as C
from TEX_Wrangle.tex_runtime import compiled_capability as _CC
from TEX_Wrangle.tex_runtime import autotier as AT


def _tiny_program():
    """A small, real TEX program + compiled type_map, shared by the CC-4/CC-5 tests
    below (mirrors test_v015_phase5.py::test_cc2_end_to_end's own fixture shape)."""
    code = "vec3 c=@A.rgb; c = c*1.3 - 0.1; c = clamp(c, 0.0, 1.0); @OUT=vec4(c,1.0);"
    bt = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}
    prog = parse_and_split(code, bt)
    tm = TypeChecker(binding_types=bt, source=code).check(prog)
    used = _collect_identifiers(prog)
    return prog, tm, used


# ── CC-3: compile_capability() ──────────────────────────────────────────────

def test_cc3_compile_capability_shape(r: SubTestResult):
    print("\n--- CC-3: compile_capability() shape ---")
    try:
        cap = C.compile_capability()
        assert set(cap.keys()) == {"cuda_inductor", "cpu_inductor", "reason"}, cap.keys()
        assert isinstance(cap["cuda_inductor"], bool)
        assert isinstance(cap["cpu_inductor"], bool)
        assert isinstance(cap["reason"], dict)
        for key, msg in cap["reason"].items():
            assert key in ("cuda_inductor", "cpu_inductor"), key
            assert isinstance(msg, str) and msg, "a reason must be a non-empty string"
            assert cap[key] is False, f"a reason exists for {key} but {key} reads True"
        for key in ("cuda_inductor", "cpu_inductor"):
            if cap[key] is False:
                assert key in cap["reason"], f"{key} is False but carries no reason"
        r.ok(f"compile_capability() shape ok on this box: {cap}")
    except Exception as e:
        r.fail("compile_capability shape", str(e))

    # Repeated calls agree, and the returned dict is a defensive copy (mutating one
    # call's result must never corrupt the memoized cache another call reads back).
    try:
        cap_a = C.compile_capability()
        cap_b = C.compile_capability()
        assert cap_a == cap_b
        cap_b["reason"]["poison"] = "x"
        cap_c = C.compile_capability()
        assert "poison" not in cap_c["reason"], "mutating a returned dict corrupted the cache"
        r.ok("compile_capability() is idempotent and returns a defensive copy")
    except Exception as e:
        r.fail("compile_capability defensive copy", str(e))


def test_cc3_capability_probed_once_and_reflects_the_probes(r: SubTestResult):
    print("\n--- CC-3: probed ONCE per process, never by compiling ---")
    # SPLIT-47 (TRK-210): `compile_capability`/`_probe_cuda_inductor`/`_probe_cpu_inductor`
    # all moved together to `compiled_capability.py`, so `compile_capability`'s internal
    # bare-name calls to the two probes now resolve through THAT module's own globals, not
    # `compiled.py`'s re-exported copies (the ROUTE-45 hazard the split's own brief named:
    # a moved function that is spied must still be looked up through the module the spy
    # patches). Patch `_CC` (compiled_capability), not `C` (compiled), for exactly this row.
    orig_cuda, orig_cpu = _CC._probe_cuda_inductor, _CC._probe_cpu_inductor
    calls = {"cuda": 0, "cpu": 0}

    def fake_cuda():
        calls["cuda"] += 1
        return False, "no triton (test)"

    def fake_cpu():
        calls["cpu"] += 1
        return True, None

    _CC._probe_cuda_inductor = fake_cuda
    _CC._probe_cpu_inductor = fake_cpu
    C._reset_capability_cache_for_test()
    try:
        cap = C.compile_capability()
        assert cap == {"cuda_inductor": False, "cpu_inductor": True,
                       "reason": {"cuda_inductor": "no triton (test)"}}, cap
        C.compile_capability()
        C.compile_capability()
        assert calls == {"cuda": 1, "cpu": 1}, (
            f"compile_capability() must probe at most once per process, got {calls}")
        r.ok("compile_capability() probes exactly once and caches across repeated calls")
    except Exception as e:
        r.fail("compile_capability probed-once contract", str(e))
    finally:
        _CC._probe_cuda_inductor, _CC._probe_cpu_inductor = orig_cuda, orig_cpu
        C._reset_capability_cache_for_test()


# ── CC-4: toolchain-aware "auto" ─────────────────────────────────────────────

def test_cc4_no_toolchain_makes_no_compile_attempt(r: SubTestResult):
    print("\n--- CC-4: absent toolchain -> auto makes NO compile attempt ---")
    prog, tm, used = _tiny_program()
    img = make_img(1, 12, 12, 3, seed=5)
    fp = "cc4_test_fp"

    ref = Interpreter().execute(prog, {"A": img}, tm, device="cpu",
                                output_names=["OUT"])["OUT"]

    submit_calls = {"n": 0}
    orig_submit = C._submit_bg_compile

    def spy_submit(*a, **kw):
        submit_calls["n"] += 1
        return orig_submit(*a, **kw)

    orig_cap = C.compile_capability_async
    C.compile_capability_async = lambda: {"cuda_inductor": False, "cpu_inductor": False,
                                    "reason": {"cuda_inductor": "test", "cpu_inductor": "test"}}
    C._submit_bg_compile = spy_submit
    AT.reset()
    try:
        ok = True
        for i in range(8):   # well past _MEASURE_COOKS=3, so the gate fires repeatedly
            out = C.run_auto(prog, {"A": img}, tm, "cpu", fp,
                             output_names=["OUT"], used_builtins=used)
            t = out["OUT"] if isinstance(out, dict) else out
            if (t - ref).abs().max().item() >= 1e-4:
                ok = False
                break
        assert ok, "run_auto output diverged from the interpreter under a no-toolchain capability"

        sp = C._consensus_extent({"A": img}, prog)
        key = AT.make_key(fp, "cpu", "fp32", sp)
        assert AT.verdict(key) == AT.REJECTED, AT.verdict(key)
        assert submit_calls["n"] == 0, (
            f"auto attempted {submit_calls['n']} compile(s) despite an absent toolchain")
        r.ok("toolchain-aware auto: 0 compile attempts, REJECTED, output stays codegen-identical")
    except Exception as e:
        r.fail("CC-4 no-toolchain gate", str(e))
    finally:
        C.compile_capability_async = orig_cap
        C._submit_bg_compile = orig_submit
        AT.reset()


def test_cc4_present_toolchain_still_submits(r: SubTestResult):
    print("\n--- CC-4: present toolchain -> the gate stays out of the way ---")
    prog, tm, used = _tiny_program()
    img = make_img(1, 12, 12, 3, seed=6)
    fp = "cc4_present_fp"

    submit_calls = {"n": 0}
    orig_submit = C._submit_bg_compile

    def spy_submit(*a, **kw):
        submit_calls["n"] += 1
        return True   # pretend it went in flight, without touching the real compile pool

    orig_cap = C.compile_capability_async
    C.compile_capability_async = lambda: {"cuda_inductor": True, "cpu_inductor": True, "reason": {}}
    C._submit_bg_compile = spy_submit
    AT.reset()
    try:
        for _ in range(3):   # exactly _MEASURE_COOKS
            C.run_auto(prog, {"A": img}, tm, "cpu", fp, output_names=["OUT"], used_builtins=used)
        assert submit_calls["n"] == 1, (
            f"expected exactly one submission once the toolchain reads present, got {submit_calls['n']}")
        sp = C._consensus_extent({"A": img}, prog)
        key = AT.make_key(fp, "cpu", "fp32", sp)
        assert AT.verdict(key) == AT.COMPILING, AT.verdict(key)
        r.ok("present toolchain: CC-4's gate does not block the existing submission path")
    except Exception as e:
        r.fail("CC-4 present-toolchain passthrough", str(e))
    finally:
        C.compile_capability_async = orig_cap
        C._submit_bg_compile = orig_submit
        AT.reset()


# ── CC-5: the lazy first-call cost lands on the background worker ──────────

@pytest.mark.slow
def test_cc5_lazy_first_call_never_stalls_the_cook_thread(r: SubTestResult):
    """A monkeypatched fake compiled fn whose FIRST invocation sleeps 2s stands in for
    torch.compile's own lazy first-call cost (Dynamo trace + Inductor/Triton lowering —
    `_try_compile` only wraps; the trace/lowering only happens once the wrapped callable
    is actually invoked). With CC-5's fix, `_submit_bg_compile`'s `warm_call` pays that
    2s on the background pool worker, so every `run_auto()` call from here (each one a
    cook tick) must complete fast -- none of them may be the thread that eats the sleep."""
    print("\n--- CC-5: lazy first-call cost never lands on the cook thread ---")
    prog, tm, used = _tiny_program()
    img = make_img(1, 16, 16, 3, seed=11)
    fp = "cc5_test_fp"
    cache_key = (fp, "cpu", "fp32")

    calls = {"n": 0}

    def fake_compiled_fn(program, bindings, type_map, device, latent_channel_count,
                         output_names, scale=None):
        calls["n"] += 1
        if calls["n"] == 1:
            time.sleep(2.0)   # stands in for torch.compile's lazy first-call trace
        names = output_names or ["OUT"]
        return {name: bindings["A"] for name in names}

    def fake_try_compile(device_type, program, type_map, **kw):
        return fake_compiled_fn, "inductor"

    orig_try_compile = C._try_compile
    orig_cap = C.compile_capability_async
    C._try_compile = fake_try_compile
    C.compile_capability_async = lambda: {"cuda_inductor": True, "cpu_inductor": True, "reason": {}}
    AT.reset()
    C._compiled_cache.pop(cache_key, None)
    C._bg_futures.pop(cache_key, None)
    try:
        max_tick_ms = 0.0
        v = None
        for _ in range(120):
            t0 = time.perf_counter()
            C.run_auto(prog, {"A": img}, tm, "cpu", fp, output_names=["OUT"], used_builtins=used)
            tick_ms = (time.perf_counter() - t0) * 1000.0
            max_tick_ms = max(max_tick_ms, tick_ms)
            sp = C._consensus_extent({"A": img}, prog)
            v = AT.verdict(AT.make_key(fp, "cpu", "fp32", sp))
            if v in (AT.COMMITTED, AT.REJECTED):
                break
            time.sleep(0.05)

        assert v in (AT.COMMITTED, AT.REJECTED), f"never reached a terminal verdict (stuck at {v})"
        assert max_tick_ms < 500.0, (
            f"a cook tick took {max_tick_ms:.1f}ms -- the lazy compile leaked onto the cook thread")
        assert calls["n"] >= 2, (
            f"expected >=2 compiled-fn calls (1 background warm-up + >=1 trial/committed), got {calls['n']}")
        r.ok(f"max cook tick {max_tick_ms:.1f}ms across {calls['n']} compiled-fn calls; verdict={v}")
    except Exception as e:
        r.fail("CC-5 no cook-thread stall", str(e))
    finally:
        C._try_compile = orig_try_compile
        C.compile_capability_async = orig_cap
        C._compiled_cache.pop(cache_key, None)
        C._bg_futures.pop(cache_key, None)
        AT.reset()


def test_cc5_warm_call_failure_discards_the_artifact(r: SubTestResult):
    """A warm_call that raises (the artifact's first real call crashes) must be treated
    exactly like a wrap failure: _bg_status reports "failed" and the artifact is not left
    in _compiled_cache for a later TRIAL to pick up broken."""
    print("\n--- CC-5: a raising warm_call discards the artifact ---")
    cache_key = ("cc5_fail_fp", "cpu", "fp32")
    C._compiled_cache.pop(cache_key, None)
    C._bg_futures.pop(cache_key, None)

    def fake_try_compile(device_type, program, type_map, **kw):
        return (lambda *a, **k: None), "inductor"

    def _boom():
        raise RuntimeError("simulated first-call compile failure")

    orig_try_compile = C._try_compile
    C._try_compile = fake_try_compile
    try:
        ok = C._submit_bg_compile(cache_key, object(), {}, "cpu", None, "fp32",
                                  "cc5_fail_fp", warm_call=_boom)
        assert ok, "submission itself (queuing the job) must still succeed"
        fut = C._bg_futures[cache_key]
        fut.result(timeout=10)
        status = C._bg_status(cache_key)
        assert status == "failed", status
        assert cache_key not in C._compiled_cache, "a broken artifact must not be left cached"
        r.ok("a raising warm_call reports 'failed' and never leaves a broken artifact cached")
    except Exception as e:
        r.fail("CC-5 warm_call failure handling", str(e))
    finally:
        C._try_compile = orig_try_compile
        C._compiled_cache.pop(cache_key, None)
        C._bg_futures.pop(cache_key, None)


# ── CC-6: bounded trial convergence ─────────────────────────────────────────

def test_cc6_convergence_bound_state_machine(r: SubTestResult):
    print("\n--- CC-6: enforce_convergence_bound state machine ---")
    try:
        AT.reset()
        k = AT.make_key("cc6_fp", "cpu", "fp32", (1, 64, 64))
        # Not yet eligible (no interp samples at all) -- the bound never applies.
        assert AT.enforce_convergence_bound(k) is False
        for _ in range(3):
            AT.record_interp(k, 5.0)
        assert AT.should_submit_compile(k) is True   # sets ready_wall, once
        assert AT._get(k).ready_wall is not None
        # Freshly eligible: nowhere near the bound yet.
        assert AT.enforce_convergence_bound(k) is False
        AT.mark_submitted(k)
        assert AT.verdict(k) == AT.COMPILING
        # Simulate elapsed wall time WITHOUT a real sleep.
        AT._get(k).ready_wall -= (AT._CONVERGENCE_BOUND_S + 1.0)
        assert AT.enforce_convergence_bound(k) is True
        assert AT.verdict(k) == AT.REJECTED
        # A terminal verdict is never re-touched by the bound (idempotent).
        assert AT.enforce_convergence_bound(k) is False
        r.ok("a key past the bound is forced REJECTED exactly once; terminal states are left alone")
    except Exception as e:
        r.fail("CC-6 state machine", str(e))


def test_cc6_fast_key_never_bound_rejected(r: SubTestResult):
    print("\n--- CC-6: a key that resolves fast is never touched by the bound ---")
    try:
        AT.reset()
        k = AT.make_key("cc6_fast", "cpu", "fp32", (1, 64, 64))
        for _ in range(3):
            AT.record_interp(k, 10.0)
        AT.should_submit_compile(k)
        AT.mark_submitted(k)
        AT.mark_ready(k)
        AT.record_trial(k, 4.0)   # a fast, clear win
        assert AT.verdict(k) == AT.COMMITTED
        assert AT.enforce_convergence_bound(k) is False, (
            "a terminal COMMITTED verdict must never be overturned by the bound")
        r.ok("a fast-resolving key commits normally; the bound never fires for it")
    except Exception as e:
        r.fail("CC-6 fast key unaffected", str(e))


def test_cc6_backlog_all_keys_eventually_terminal(r: SubTestResult):
    """Models a multi-stage playback program in miniature: many keys become ELIGIBLE
    (3 interp samples) but a gate (headroom/capture-in-flight, or a busy single-worker
    queue) keeps saying no, so should_submit_compile never gets a successful submission
    through for most of them. Without a bound those keys read "measuring" forever; with
    it, every one of them reaches a terminal verdict once its own ready_wall clock runs
    out -- proven here without any real sleep."""
    print("\n--- CC-6: a whole backlog of keys reaches a terminal verdict, none stuck ---")
    try:
        AT.reset()
        keys = [AT.make_key(f"cc6_stage{i}", "cpu", "fp32", (1, 2048, 2048)) for i in range(10)]
        for k in keys:
            for _ in range(3):
                AT.record_interp(k, 5.0)
            assert AT.should_submit_compile(k) is True   # eligible, but never actually submitted
            assert AT.verdict(k) == AT.MEASURING, "still MEASURING until it is EITHER submitted or bounded"
        # Time passes; the gate keeps saying no for all ten the whole time.
        for k in keys:
            AT._get(k).ready_wall -= (AT._CONVERGENCE_BOUND_S + 0.5)
        fired = [AT.enforce_convergence_bound(k) for k in keys]
        assert all(fired), fired
        assert all(AT.verdict(k) == AT.REJECTED for k in keys)
        r.ok("all ten backlogged keys reached a terminal verdict at the bound, none stuck 'measuring'")
    except Exception as e:
        r.fail("CC-6 backlog convergence", str(e))


def test_cc6_wired_into_run_auto(r: SubTestResult):
    """Integration: run_auto itself calls enforce_convergence_bound and routes to
    codegen (never crashing, never re-entering the compile machinery) once a key's
    ready_wall is stale -- exercised by rewinding time rather than by actually waiting
    _CONVERGENCE_BOUND_S seconds. `_try_compile` is monkeypatched to an instant fake (no
    sleep, no real torch.compile) so this stays a fast, deterministic test of the
    WIRING, not a repeat of CC-5's own timing proof."""
    print("\n--- CC-6: run_auto honours the convergence bound ---")
    prog, tm, used = _tiny_program()
    img = make_img(1, 10, 10, 3, seed=9)
    fp = "cc6_run_auto_fp"
    cache_key = (fp, "cpu", "fp32")

    ref = Interpreter().execute(prog, {"A": img}, tm, device="cpu",
                                output_names=["OUT"])["OUT"]

    def fake_compiled_fn(program, bindings, type_map, device, latent_channel_count,
                         output_names, scale=None):
        names = output_names or ["OUT"]
        return {name: bindings["A"] for name in names}

    def fake_try_compile(device_type, program, type_map, **kw):
        return fake_compiled_fn, "inductor"

    orig_try_compile = C._try_compile
    orig_cap = C.compile_capability_async
    C._try_compile = fake_try_compile
    C.compile_capability_async = lambda: {"cuda_inductor": True, "cpu_inductor": True, "reason": {}}
    AT.reset()
    C._compiled_cache.pop(cache_key, None)
    C._bg_futures.pop(cache_key, None)
    try:
        for _ in range(3):
            C.run_auto(prog, {"A": img}, tm, "cpu", fp, output_names=["OUT"], used_builtins=used)
        sp = C._consensus_extent({"A": img}, prog)
        key = AT.make_key(fp, "cpu", "fp32", sp)
        assert AT.verdict(key) in (AT.COMPILING, AT.MEASURING, AT.TRIAL), AT.verdict(key)
        st = AT._get(key)
        assert st.ready_wall is not None
        st.ready_wall -= (AT._CONVERGENCE_BOUND_S + 1.0)   # simulate elapsed time

        out = C.run_auto(prog, {"A": img}, tm, "cpu", fp, output_names=["OUT"], used_builtins=used)
        t = out["OUT"] if isinstance(out, dict) else out
        assert (t - ref).abs().max().item() < 1e-4, "bound-rejected cook must still be codegen-correct"
        assert AT.verdict(key) == AT.REJECTED, AT.verdict(key)
        r.ok("run_auto forces REJECTED at the bound and keeps serving correct pixels")
    except Exception as e:
        r.fail("CC-6 run_auto wiring", str(e))
    finally:
        C._try_compile = orig_try_compile
        C.compile_capability_async = orig_cap
        C._compiled_cache.pop(cache_key, None)
        C._bg_futures.pop(cache_key, None)
        AT.reset()


# ── CC-7 (AUTO-47): the toolchain probe itself never blocks the cook thread ─────────

@pytest.mark.slow
def test_cc7_capability_probe_never_blocks_the_cook_thread(r: SubTestResult):
    """Measured on an embedding host (a GPU box whose venv has no Triton):
    `compile_capability()`'s CPU-toolchain probe (`_probe_cpu_inductor`'s Windows vcvarsall
    glob + subprocess) measured 1.4-1.8s
    on a quiet box, once 9.6s under GPU contention -- squarely on the cook thread, at the
    exact moment a key first becomes eligible to compile (`should_submit_compile`).
    Simulates that cost with a monkeypatched slow probe (no real vcvarsall search, so this
    stays CPU-only and fast even when it passes) and proves every `run_auto` tick from
    here stays fast regardless -- the probe now runs on its own background thread
    (`compile_capability_async`); a tick that catches it still pending just leaves the key
    MEASURING for the next tick to re-check, exactly like a busy compile-pool or a failed
    headroom check already do for other reasons. Red at base: before AUTO-47's fix,
    `run_auto` called the blocking `compile_capability()` inline here, so the one tick
    that first reaches this branch took >=2s."""
    print("\n--- CC-7: the capability probe never stalls a cook tick ---")
    prog, tm, used = _tiny_program()
    img = make_img(1, 12, 12, 3, seed=13)
    fp = "cc7_test_fp"

    def slow_cpu_probe():
        time.sleep(2.0)
        return False, "no compiler (test, slowed to simulate a real vcvarsall search)"

    def fast_no_cuda_probe():
        return False, "no cuda (test)"

    orig_cpu = _CC._probe_cpu_inductor
    orig_cuda = _CC._probe_cuda_inductor
    _CC._probe_cpu_inductor = slow_cpu_probe
    _CC._probe_cuda_inductor = fast_no_cuda_probe
    _CC._reset_capability_cache_for_test()
    AT.reset()
    try:
        max_tick_ms = 0.0
        verdict = None
        for _ in range(200):
            t0 = time.perf_counter()
            C.run_auto(prog, {"A": img}, tm, "cpu", fp, output_names=["OUT"], used_builtins=used)
            tick_ms = (time.perf_counter() - t0) * 1000.0
            max_tick_ms = max(max_tick_ms, tick_ms)
            sp = C._consensus_extent({"A": img}, prog)
            verdict = AT.verdict(AT.make_key(fp, "cpu", "fp32", sp))
            if verdict == AT.REJECTED:
                break
            time.sleep(0.02)

        assert verdict == AT.REJECTED, f"never reached a terminal verdict (stuck at {verdict})"
        assert max_tick_ms < 500.0, (
            f"a cook tick took {max_tick_ms:.1f}ms -- the capability probe leaked onto "
            f"the cook thread")
        r.ok(f"max cook tick {max_tick_ms:.1f}ms while a 2s probe ran in the background; "
             f"verdict={verdict}")
    except Exception as e:
        r.fail("CC-7 capability probe never blocks the cook thread", str(e))
    finally:
        _CC._probe_cpu_inductor = orig_cpu
        _CC._probe_cuda_inductor = orig_cuda
        _CC._reset_capability_cache_for_test()
        AT.reset()


def test_cc7_async_returns_none_then_the_real_answer(r: SubTestResult):
    """Unit-level (no run_auto): `compile_capability_async()` returns `None` on every call
    before the background probe finishes, then the SAME dict `compile_capability()` would
    give, and never re-probes once resolved -- direct coverage of the function CC-7's
    integration test above exercises indirectly through run_auto."""
    print("\n--- CC-7: compile_capability_async() unit contract ---")
    calls = {"n": 0}

    def slow_cpu_probe():
        calls["n"] += 1
        time.sleep(0.3)
        return True, None

    def fast_no_cuda_probe():
        return False, "no cuda (test)"

    orig_cpu = _CC._probe_cpu_inductor
    orig_cuda = _CC._probe_cuda_inductor
    _CC._probe_cpu_inductor = slow_cpu_probe
    _CC._probe_cuda_inductor = fast_no_cuda_probe
    _CC._reset_capability_cache_for_test()
    try:
        first = _CC.compile_capability_async()
        assert first is None, f"expected None before the background probe finishes, got {first}"
        deadline = time.perf_counter() + 5.0
        result = None
        while time.perf_counter() < deadline:
            result = _CC.compile_capability_async()
            if result is not None:
                break
            time.sleep(0.02)
        assert result == {"cuda_inductor": False, "cpu_inductor": True, "reason":
                          {"cuda_inductor": "no cuda (test)"}}, result
        assert calls["n"] == 1, f"the probe must run exactly once, ran {calls['n']}"
        again = _CC.compile_capability_async()
        assert again == result and calls["n"] == 1, "a resolved probe must never re-run"
        r.ok("compile_capability_async(): None until resolved, then the real (cached) answer")
    except Exception as e:
        r.fail("CC-7 compile_capability_async unit contract", str(e))
    finally:
        _CC._probe_cpu_inductor = orig_cpu
        _CC._probe_cuda_inductor = orig_cuda
        _CC._reset_capability_cache_for_test()
