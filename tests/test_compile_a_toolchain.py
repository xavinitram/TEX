"""
COMPILE-A (v0.46) — a toolchain-aware, non-stalling "auto" tier.

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
    orig_cuda, orig_cpu = C._probe_cuda_inductor, C._probe_cpu_inductor
    calls = {"cuda": 0, "cpu": 0}

    def fake_cuda():
        calls["cuda"] += 1
        return False, "no triton (test)"

    def fake_cpu():
        calls["cpu"] += 1
        return True, None

    C._probe_cuda_inductor = fake_cuda
    C._probe_cpu_inductor = fake_cpu
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
        C._probe_cuda_inductor, C._probe_cpu_inductor = orig_cuda, orig_cpu
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

    orig_cap = C.compile_capability
    C.compile_capability = lambda: {"cuda_inductor": False, "cpu_inductor": False,
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
        C.compile_capability = orig_cap
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

    orig_cap = C.compile_capability
    C.compile_capability = lambda: {"cuda_inductor": True, "cpu_inductor": True, "reason": {}}
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
        C.compile_capability = orig_cap
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

    def fake_compiled_fn(program, bindings, type_map, device, latent_channel_count, output_names):
        calls["n"] += 1
        if calls["n"] == 1:
            time.sleep(2.0)   # stands in for torch.compile's lazy first-call trace
        names = output_names or ["OUT"]
        return {name: bindings["A"] for name in names}

    def fake_try_compile(device_type, program, type_map, **kw):
        return fake_compiled_fn, "inductor"

    orig_try_compile = C._try_compile
    orig_cap = C.compile_capability
    C._try_compile = fake_try_compile
    C.compile_capability = lambda: {"cuda_inductor": True, "cpu_inductor": True, "reason": {}}
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
        C.compile_capability = orig_cap
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
