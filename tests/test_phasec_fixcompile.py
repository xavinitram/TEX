"""
FIX-COMPILE (v0.46 Phase C) — adversarial reproduction + regression tests for C1-C8 of
`docs/worklog/v046/phaseC/CONSOLIDATED.md`'s FIX-COMPILE section (findings B1#1-4, R3#1,
B6#6-7, R1#3). Each row below is written to be RED against the pre-fix code and GREEN
against the fix, so this file is itself the adversarial verification for every item.

C9 (optional, "lane's call") is a documentation-only decision (kept the fixed 30s
convergence bound rather than deriving it) — see the comment beside
`autotier._CONVERGENCE_BOUND_S`; there is no behaviour to test.

PORTABILITY: every row here is CPU-only and uses monkeypatched fakes for anything that
would otherwise need a real torch.compile / CUDA / MSVC toolchain — this box's Smart App
Control policy forbids looping real compiles (see test_compile_a_toolchain.py's own
portability note, which this file follows).
"""
import sys
import threading
import time

from helpers import *  # noqa: F401,F403
from TEX_Wrangle.tex_cache import parse_and_split
from TEX_Wrangle.tex_runtime import compiled as C
from TEX_Wrangle.tex_runtime import autotier as AT
from TEX_Wrangle.tex_runtime import codegen as CG
from TEX_Wrangle.tex_runtime import host as HOST
import TEX_Wrangle.tex_doctor as DOCTOR
from TEX_Wrangle import tex_testkit


def _prewarm_dynamo_import():
    """`_precompile_ctx()` lazily does `import torch._dynamo.config` on its FIRST call in
    the whole process — a genuinely slow import (hundreds of ms) that has nothing to do
    with any program being compiled. Timing-sensitive rows below must pay that cost
    BEFORE their timed section, or they conflate a one-time import tax with the behaviour
    under test."""
    try:
        import torch._dynamo.config  # noqa: F401
    except Exception:
        pass


def _tiny_program():
    code = "vec3 c=@A.rgb; c = c*1.3 - 0.1; c = clamp(c, 0.0, 1.0); @OUT=vec4(c,1.0);"
    bt = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}
    prog = parse_and_split(code, bt)
    tm = TypeChecker(binding_types=bt, source=code).check(prog)
    used = _collect_identifiers(prog)
    return prog, tm, used


# ── C1: warm_call must never share a pool with a blocking trial/committed submit ────

def test_c1_warm_never_stalls_a_different_keys_compile_pool_submit(r: SubTestResult):
    print("\n--- C1: another key's compile-wrap must not queue behind this key's warm ---")
    _prewarm_dynamo_import()
    key_a = ("c1_a_fp", "cpu", "fp32")
    key_b = ("c1_b_fp", "cpu", "fp32")
    for k in (key_a, key_b):
        C._compiled_cache.pop(k, None)
        C._bg_futures.pop(k, None)

    def fake_try_compile(device_type, program, type_map, **kw):
        return (lambda *a, **k: None), "inductor"

    def slow_warm():
        time.sleep(0.5)   # stands in for the real 10-30s lazy first-call cost

    orig_try_compile = C._try_compile
    C._try_compile = fake_try_compile
    try:
        ok_a = C._submit_bg_compile(key_a, object(), {}, "cpu", None, "fp32",
                                    "c1_a_fp", warm_call=slow_warm)
        assert ok_a, "key A's submission must itself succeed"
        # key A's warm is now (slowly) running in the background. key B's OWN
        # compile-wrap submission shares `_COMPILE_POOL` and must complete promptly —
        # it must never queue behind key A's warm.
        t0 = time.perf_counter()
        ok_b = C._submit_bg_compile(key_b, object(), {}, "cpu", None, "fp32", "c1_b_fp")
        assert ok_b, "key B's submission must itself succeed"
        fut_b = C._bg_futures[key_b]
        fut_b.result(timeout=3.0)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert elapsed_ms < 350.0, (
            f"key B's compile-wrap took {elapsed_ms:.1f}ms — it queued behind key A's "
            "0.5s warm on the SAME single-worker pool")
        r.ok(f"C1: key B's wrap finished in {elapsed_ms:.1f}ms while key A's warm was "
             "still running")
    except Exception as e:
        r.fail("C1 warm isolation", f"{type(e).__name__}: {e}")
    finally:
        C._try_compile = orig_try_compile
        for k in (key_a, key_b):
            fut = C._bg_futures.pop(k, None)
            if fut is not None:
                try:
                    fut.result(timeout=3.0)
                except Exception:
                    pass
            C._compiled_cache.pop(k, None)


def test_c1_warm_call_failure_still_discards_the_artifact(r: SubTestResult):
    """Guards against a regression in the split: a raising warm_call, now running on
    `_WARM_POOL`, must still report 'failed' through the SAME `_bg_futures` entry and
    still evict the artifact — exactly as it did when warm ran on `_COMPILE_POOL`."""
    print("\n--- C1: a raising warm_call (now on its own pool) still discards the artifact ---")
    cache_key = ("c1_fail_fp", "cpu", "fp32")
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
                                  "c1_fail_fp", warm_call=_boom)
        assert ok
        fut = C._bg_futures[cache_key]
        fut.result(timeout=5.0)
        status = C._bg_status(cache_key)
        assert status == "failed", status
        assert cache_key not in C._compiled_cache, "a broken artifact must not be left cached"
        r.ok("C1: a raising warm_call on the dedicated pool still reports 'failed'")
    except Exception as e:
        r.fail("C1 warm failure on split pool", str(e))
    finally:
        C._try_compile = orig_try_compile
        C._compiled_cache.pop(cache_key, None)
        C._bg_futures.pop(cache_key, None)


# ── C2: a test fixture must drain background warm/compile jobs ─────────────────────

def test_c2_cold_engine_state_drains_bg_futures(r: SubTestResult):
    print("\n--- C2: cold_engine_state drains an in-flight background warm job ---")
    cache_key = ("c2_test_fp", "cpu", "fp32")
    C._compiled_cache.pop(cache_key, None)
    C._bg_futures.pop(cache_key, None)

    warm_calls = {"n": 0}

    def slow_warm():
        time.sleep(0.3)
        warm_calls["n"] += 1

    def fake_try_compile(device_type, program, type_map, **kw):
        return (lambda *a, **k: None), "inductor"

    orig_try_compile = C._try_compile
    C._try_compile = fake_try_compile
    try:
        ok = C._submit_bg_compile(cache_key, object(), {}, "cpu", None, "fp32",
                                  "c2_test_fp", warm_call=slow_warm)
        assert ok
        assert cache_key in C._bg_futures, "the job must still be in flight right after submitting"
        with tex_testkit.cold_engine_state():
            pass
        assert warm_calls["n"] == 1, (
            "the warm job must have been drained (waited to completion) by the fixture, "
            f"but it ran {warm_calls['n']} times by the time the block exited")
        assert cache_key not in C._bg_futures, "a drained future must be forgotten"
        r.ok("C2: cold_engine_state waits out an in-flight background job before returning")
    except Exception as e:
        r.fail("C2 cold_engine_state drain", str(e))
    finally:
        C._try_compile = orig_try_compile
        C._compiled_cache.pop(cache_key, None)
        C._bg_futures.pop(cache_key, None)


# ── C3: fold the projected warm-clone size into the headroom check, and cap it ──────

def test_c3_headroom_folds_projected_clone_bytes(r: SubTestResult):
    print("\n--- C3: _cuda_headroom_ok folds a projected clone size into its threshold ---")
    dev = torch.device("cuda:0")   # index given -- never calls torch.cuda.current_device()

    class _FakeHost:
        def __init__(self, free_bytes):
            self._free = free_bytes

        def get_free_memory(self, _dev):
            return self._free

    orig_get_host_services = HOST.get_host_services
    try:
        # 2.5 GB free: comfortably above the flat 2GB floor, but NOT above 2GB + a 1GB clone.
        HOST.get_host_services = lambda: _FakeHost(int(2.5 * 1024 ** 3))
        assert C._cuda_headroom_ok(dev, extra_bytes=0) is True, "the plain 2GB check must still pass"
        assert C._cuda_headroom_ok(dev, extra_bytes=1 * 1024 ** 3) is False, (
            "a 1GB projected clone on a 2.5GB-free box must fail the folded headroom check")
        r.ok("C3: _cuda_headroom_ok(extra_bytes=...) folds the projected clone size in")
    except Exception as e:
        r.fail("C3 headroom folds clone bytes", f"{type(e).__name__}: {e}")
    finally:
        HOST.get_host_services = orig_get_host_services


def test_c3_warm_skipped_above_clone_cap(r: SubTestResult):
    print("\n--- C3: a projected clone above the cap skips the warm submission entirely ---")
    prog, tm, used = _tiny_program()
    img = make_img(1, 12, 12, 3, seed=21)
    fp = "c3_cap_fp"
    cache_key = (fp, "cpu", "fp32")
    C._compiled_cache.pop(cache_key, None)
    C._bg_futures.pop(cache_key, None)

    orig_cap_bytes = C._WARM_CLONE_CAP_BYTES
    orig_cap_fn = C.compile_capability
    orig_submit = C._submit_bg_compile
    submit_calls = {"n": 0}

    def spy_submit(*a, **kw):
        submit_calls["n"] += 1
        return True

    C.compile_capability = lambda: {"cuda_inductor": True, "cpu_inductor": True, "reason": {}}
    C._WARM_CLONE_CAP_BYTES = 8   # tiny -- this program's image binding blows past it
    C._submit_bg_compile = spy_submit
    AT.reset()
    try:
        for _ in range(3):   # _MEASURE_COOKS
            C.run_auto(prog, {"A": img}, tm, "cpu", fp, output_names=["OUT"], used_builtins=used)
        assert submit_calls["n"] == 0, (
            f"expected 0 warm submissions above the clone cap, got {submit_calls['n']}")
        r.ok("C3: a binding set whose clone exceeds the cap never reaches _submit_bg_compile")
    except Exception as e:
        r.fail("C3 clone cap skip", str(e))
    finally:
        C._WARM_CLONE_CAP_BYTES = orig_cap_bytes
        C.compile_capability = orig_cap_fn
        C._submit_bg_compile = orig_submit
        C._compiled_cache.pop(cache_key, None)
        C._bg_futures.pop(cache_key, None)
        AT.reset()


# ── C4: never persist a toolchain-absent REJECTED ───────────────────────────────────

def test_c4_toolchain_absent_rejection_not_persisted(r: SubTestResult):
    print("\n--- C4: a toolchain-absent REJECTED verdict must not survive a reload ---")
    with cold_engine_state():
        prog, tm, used = _tiny_program()
        img = make_img(1, 12, 12, 3, seed=22)
        fp = "c4_test_fp"
        orig_cap = C.compile_capability
        C.compile_capability = lambda: {"cuda_inductor": False, "cpu_inductor": False,
                                        "reason": {"cuda_inductor": "t", "cpu_inductor": "t"}}
        try:
            for _ in range(5):
                C.run_auto(prog, {"A": img}, tm, "cpu", fp, output_names=["OUT"], used_builtins=used)
            sp = C._consensus_extent({"A": img}, prog)
            key = AT.make_key(fp, "cpu", "fp32", sp)
            assert AT.verdict(key) == AT.REJECTED, AT.verdict(key)
            # Drop the in-memory record and reload straight from disk: if C4's fix holds,
            # the toolchain-absent rejection was never written, so nothing comes back.
            AT._STATE.pop(key, None)
            AT.reload()
            assert key not in AT._STATE, (
                "a toolchain-absent REJECTED was persisted and reloaded as terminal -- it "
                "would never re-measure even after the toolchain becomes available")
            r.ok("C4: a toolchain-absent REJECTED verdict is kept in-memory only")
        except Exception as e:
            r.fail("C4 toolchain-absent not persisted", str(e))
        finally:
            C.compile_capability = orig_cap


# ── C5: evict a bound-rejected key's artifact from _compiled_cache ──────────────────

def test_c5_convergence_bound_evicts_compiled_cache(r: SubTestResult):
    """Models the window B1#4 describes: the background job already wrote the artifact
    into `_compiled_cache` (a real compile finishes there before anything polls
    `_bg_status` to promote autotier's own verdict from COMPILING to TRIAL), and THEN the
    bound fires because `ready_wall` is stale. `state == TRIAL` with the artifact already
    present is a DIFFERENT, earlier branch in `run_auto` (a normal timed trial) and never
    reaches the bound at all in that cook — COMPILING-with-a-ready-artifact is the branch
    that actually reaches `enforce_convergence_bound` while `_compiled_cache` is occupied."""
    print("\n--- C5: a convergence-bound rejection evicts the key's cached artifact ---")
    with cold_engine_state():
        prog, tm, used = _tiny_program()
        img = make_img(1, 10, 10, 3, seed=23)
        fp = "c5_test_fp"
        cache_key = (fp, "cpu", "fp32")
        sp = C._consensus_extent({"A": img}, prog)
        key = AT.make_key(fp, "cpu", "fp32", sp)
        for _ in range(3):
            AT.record_interp(key, 5.0)
        AT.should_submit_compile(key)
        AT.mark_submitted(key)   # -> COMPILING
        C._compiled_cache[cache_key] = (lambda *a, **k: None, "inductor")
        C._verify_state[cache_key] = {"px": 100, "samples": []}
        AT._get(key).ready_wall -= (AT._CONVERGENCE_BOUND_S + 1.0)
        try:
            assert AT.verdict(key) == AT.COMPILING, AT.verdict(key)
            C.run_auto(prog, {"A": img}, tm, "cpu", fp, output_names=["OUT"], used_builtins=used)
            assert AT.verdict(key) == AT.REJECTED, AT.verdict(key)
            assert cache_key not in C._compiled_cache, (
                "the bound rejected this key but left its artifact cached, wasting an LRU slot")
            assert cache_key not in C._verify_state
            r.ok("C5: a convergence-bound rejection evicts _compiled_cache/_verify_state")
        except Exception as e:
            r.fail("C5 convergence bound eviction", str(e))
        finally:
            C._compiled_cache.pop(cache_key, None)
            C._verify_state.pop(cache_key, None)


# ── C6: _setup_msvc_env must not flip its flag before the work completes ───────────

def test_c6_setup_msvc_env_lock_orders_completion_before_flag(r: SubTestResult):
    print("\n--- C6: a concurrent caller waits for the in-progress setup, never reads stale state ---")
    if sys.platform != "win32":
        r.ok("C6: non-Windows -- _setup_msvc_env is a no-op, nothing to race")
        return

    saved_flag = C._msvc_env_initialized
    saved_include = os.environ.get("INCLUDE")
    saved_path = os.environ.get("PATH")
    C._msvc_env_initialized = False
    os.environ.pop("INCLUDE", None)

    started = threading.Event()
    finish_gate = threading.Event()

    def fake_glob(pattern, recursive=False):
        started.set()
        finish_gate.wait(timeout=5)
        return ["C:\\fake\\vcvarsall.bat"] if "vcvarsall" in pattern else []

    def fake_run(*a, **kw):
        class _R:
            returncode = 0
            stdout = "INCLUDE=C:\\fake_include\n"
        return _R()

    orig_glob, orig_run = C.glob.glob, C.subprocess.run
    C.glob.glob = fake_glob
    C.subprocess.run = fake_run
    results = {}
    try:
        def worker_a():
            C._setup_msvc_env()

        def worker_b():
            started.wait(timeout=5)
            C._setup_msvc_env()
            results["B_INCLUDE"] = os.environ.get("INCLUDE")

        ta = threading.Thread(target=worker_a)
        tb = threading.Thread(target=worker_b)
        ta.start()
        tb.start()
        time.sleep(0.2)   # let B actually call _setup_msvc_env and either block or return
        finish_gate.set()
        ta.join(timeout=5)
        tb.join(timeout=5)
        assert results.get("B_INCLUDE") == "C:\\fake_include", (
            f"thread B observed INCLUDE={results.get('B_INCLUDE')!r} — it read the "
            "environment before A's setup had actually finished")
        r.ok("C6: a concurrent caller blocks until the in-progress setup completes")
    except Exception as e:
        r.fail("C6 msvc env lock ordering", f"{type(e).__name__}: {e}")
    finally:
        C.glob.glob, C.subprocess.run = orig_glob, orig_run
        C._msvc_env_initialized = saved_flag
        if saved_include is None:
            os.environ.pop("INCLUDE", None)
        else:
            os.environ["INCLUDE"] = saved_include
        if saved_path is not None:
            os.environ["PATH"] = saved_path


# ── C7: _ES_CO_MEMO's check-then-set has no lock ────────────────────────────────────

def test_c7_es_co_memo_setdefault_race(r: SubTestResult):
    print("\n--- C7: two threads racing on the same device converge on ONE _es object ---")
    device = torch.device("cpu")
    barrier = threading.Barrier(2, timeout=5)

    class _SlowGetDict(dict):
        """A plain dict's bound `get` can't be monkeypatched (it's a read-only slot on
        the builtin type), so force the race deterministically with a dict SUBCLASS
        instead: both racing threads must clear the barrier inside `.get()` — i.e. both
        must have already read `None` — before either is allowed to proceed to its
        `setdefault` write."""

        def get(self, key, default=None):
            result = dict.get(self, key, default)
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                pass
            return result

    orig_memo = CG._ES_CO_MEMO
    CG._ES_CO_MEMO = _SlowGetDict()

    results = []
    lock = threading.Lock()

    def worker():
        es = CG._get_es_co(device)
        with lock:
            results.append(es)

    try:
        t1 = threading.Thread(target=worker)
        t2 = threading.Thread(target=worker)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)
        assert len(results) == 2, f"expected 2 results, got {len(results)}"
        assert results[0] is results[1], (
            "the two threads returned DIFFERENT _es objects for the same device — "
            "torch.compile would see a changed callable identity across cooks")
        assert len(CG._ES_CO_MEMO) == 1 or device in CG._ES_CO_MEMO, CG._ES_CO_MEMO
        r.ok("C7: setdefault makes every racing caller converge on the same _es object")
    except Exception as e:
        r.fail("C7 _ES_CO_MEMO race", f"{type(e).__name__}: {e}")
    finally:
        CG._ES_CO_MEMO = orig_memo


# ── C8: tex_doctor must delegate its CUDA probe, not duplicate it ──────────────────

def test_c8_doctor_delegates_cuda_probe_to_compiled(r: SubTestResult):
    print("\n--- C8: tex_doctor._inductor_prereq('cuda') delegates to compiled._probe_cuda_inductor ---")
    orig = C._probe_cuda_inductor
    calls = {"n": 0}

    def fake():
        calls["n"] += 1
        return False, "fake reason from compiled"

    C._probe_cuda_inductor = fake
    try:
        result = DOCTOR._inductor_prereq("cuda")
        assert calls["n"] == 1, (
            "tex_doctor._inductor_prereq('cuda') must call compiled._probe_cuda_inductor "
            f"exactly once, got {calls['n']} calls -- it may be reimplementing the probe")
        assert result == (False, "fake reason from compiled"), result
        r.ok("C8: the doctor's CUDA row is compiled._probe_cuda_inductor's own answer")
    except Exception as e:
        r.fail("C8 doctor delegates cuda probe", f"{type(e).__name__}: {e}")
    finally:
        C._probe_cuda_inductor = orig
