"""FIX-COMPILE (v0.50.0 Phase C) — K1-K6, red-first per item. Each test below
reproduces its finding against the pre-fix shape directly (a local monkeypatch/
stand-in reconstructing the old behaviour, or a direct exercise of the real fix)
rather than requiring a checkout of the base commit, so the file is its own
red-first evidence.

PORTABILITY: CPU-only (no CUDA/MSVC/Triton assumed); K1 uses a real `torch.compile(...,
backend="aot_eager")` (no Inductor/toolchain needed for aot_eager) — the same backend
COMPILETRY-50's own D2 test uses for the identical reason.
"""
import threading
import time
import types

import torch
import torch._dynamo.config as _dynamo_config

from helpers import *  # noqa: F401,F403
from TEX_Wrangle.tex_cache import parse_and_split
from TEX_Wrangle.tex_runtime import codegen_persist as CP
from TEX_Wrangle.tex_runtime import fncalls_compile as FC
from TEX_Wrangle.tex_runtime import compiled as C


def _spatial_program():
    """A program with enough tensor ops / spatial context to clear
    `execute_compiled`'s own early gates (op_count>=8, loop_depth<=2, has_spatial)."""
    code = ("vec3 c=@A.rgb; c = c*1.3 - 0.1; c = c + 0.05; c = c*0.9; "
           "c = clamp(c, 0.0, 1.0); c = pow(c, 1.1); c = c*c; "
           "@OUT=vec4(c,1.0);")
    bt = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}
    prog = parse_and_split(code, bt)
    tm = TypeChecker(binding_types=bt, source=code).check(prog)
    used = _collect_identifiers(prog)
    return prog, tm, used


# ── K1: codegen's exec-globals lacked __name__ -> Dynamo KeyError on a graph-break
# resume. Fix: a REAL per-build module registered in sys.modules (_codegen_exec_namespace),
# not a bare dict and not a synthetic placeholder string. ──────────────────────────────

_K1_HELPER_SRC = """
import torch as _torch

@_torch.compiler.disable()
def _helper(x):
    return x + 1.0
"""

_K1_FN_SRC = """
def _tex_fn(x):
    y = x * 2.0
    y = _helper(y)
    z = y * 3.0
    return z
"""


def _k1_build(namespace: dict):
    exec(compile(_K1_HELPER_SRC, "<k1-helper>", "exec"), namespace)
    exec(compile(_K1_FN_SRC, "<k1-fn>", "exec"), namespace)
    return namespace["_tex_fn"]


def test_k1_bare_dict_globals_raise_keyerror_on_resume(r: SubTestResult):
    """RED-FIRST EVIDENCE: the exact pre-fix shape (a bare dict with no dunders, the
    literal `{"_MF": ..., "_CK": ..., "_SCM": ...}` `codegen.py`/`codegen_persist.py`
    used to hand `exec()`) really does raise `KeyError: '__name__'` inside Dynamo's
    graph-break resume, under `caching_precompile=True` -- the exact flag
    `compiled._precompile_ctx()` sets around every real compile attempt."""
    print("\n--- K1 (red-first): a bare exec-globals dict raises on a graph-break resume ---")
    fn = _k1_build({})
    try:
        with _dynamo_config.patch(caching_precompile=True):
            cfn = torch.compile(fn, backend="aot_eager", fullgraph=False)
            cfn(torch.ones(3))
        r.fail("K1 red-first", "expected a KeyError on '__name__' from the bare-dict "
               "globals shape, but the call succeeded -- the red-first repro no longer "
               "reproduces the defect this fix targets")
    except Exception as e:
        if "__name__" in str(e) or "__name__" in type(e).__name__:
            r.ok(f"K1 red-first confirmed: bare-dict globals raise "
                 f"{type(e).__name__} mentioning __name__, exactly B3#1's finding")
        else:
            r.fail("K1 red-first", f"raised {type(e).__name__}: {e!r} -- not the "
                   "expected __name__ KeyError shape")
    finally:
        torch._dynamo.reset()


def test_k1_codegen_exec_namespace_survives_a_graph_break_resume(r: SubTestResult):
    """GREEN: the real fix, `codegen_persist._codegen_exec_namespace`, used exactly the
    way `codegen.py`'s `_CodeGen.build()` now uses it -- a real module in sys.modules,
    not a bare dict and not a synthetic unregistered string (B3#1's own middle finding:
    that trades the KeyError for ModuleNotFoundError)."""
    print("\n--- K1: _codegen_exec_namespace's real module survives a graph-break resume ---")
    filename = "<tex_codegen_k1testfixture>"
    try:
        namespace = CP._codegen_exec_namespace(filename, {})
        mod_name = CP._codegen_module_name(filename)
        assert mod_name in __import__("sys").modules, (
            "the fix must register a REAL module in sys.modules, found none")
        assert __import__("sys").modules[mod_name].__dict__ is namespace, (
            "the returned namespace must BE the registered module's own __dict__")
        fn = _k1_build(namespace)
        with _dynamo_config.patch(caching_precompile=True):
            cfn = torch.compile(fn, backend="aot_eager", fullgraph=False)
            out1 = cfn(torch.ones(3))
            out2 = cfn(torch.ones(3) * 2)   # a second call, same guard -- exercises resume reuse
        assert torch.equal(out1, torch.tensor([9.0, 9.0, 9.0]))
        assert torch.equal(out2, torch.tensor([15.0, 15.0, 15.0]))
        r.ok("K1: a real per-build module survives torch.compile's graph-break resume "
             "under caching_precompile, with correct results")
    except Exception as e:
        r.fail("K1 codegen_exec_namespace", f"{type(e).__name__}: {e}")
    finally:
        torch._dynamo.reset()
        __import__("sys").modules.pop(CP._codegen_module_name(filename), None)
        import linecache
        linecache.cache.pop(filename, None)
        try:
            CP._LINECACHE_KEYS.remove(filename)
        except ValueError:
            pass


def test_k1_module_is_bounded_and_evicted_with_its_linecache_entry(r: SubTestResult):
    """K1's own bound: the synthetic module is evicted from sys.modules in lockstep with
    its paired linecache entry, so a long session cannot leak one module per fingerprint
    forever (B3#1's own stated risk for the 'point at a real, shared module' alternative
    this fix deliberately avoids)."""
    print("\n--- K1: the synthetic module is evicted alongside its linecache entry ---")
    import sys as _sys
    saved_keys = list(CP._LINECACHE_KEYS)
    saved_max = CP._LINECACHE_MAX
    CP._LINECACHE_MAX = 2
    names = [f"<tex_codegen_k1bound{i}>" for i in range(4)]
    try:
        for n in names:
            CP._register_codegen_linecache(n, "def _tex_fn():\n    return 1\n")
            CP._codegen_exec_namespace(n, {})
        # only the last _LINECACHE_MAX filenames should still have BOTH a linecache
        # entry and a registered module -- the earlier ones must have been evicted
        # from both together.
        still_present = [n for n in names if n in CP._LINECACHE_KEYS]
        assert len(still_present) == CP._LINECACHE_MAX, still_present
        for n in names:
            mod_name = CP._codegen_module_name(n)
            in_linecache = n in __import__("linecache").cache
            in_modules = mod_name in _sys.modules
            assert in_linecache == in_modules, (
                f"{n}: linecache present={in_linecache} but sys.modules present="
                f"{in_modules} -- the two registries drifted apart")
        r.ok(f"K1: linecache and sys.modules stay in lockstep across eviction "
             f"({len(names)} builds, cap {CP._LINECACHE_MAX})")
    except Exception as e:
        r.fail("K1 bounded eviction", f"{type(e).__name__}: {e}")
    finally:
        for n in names:
            _sys.modules.pop(CP._codegen_module_name(n), None)
            __import__("linecache").cache.pop(n, None)
        CP._LINECACHE_MAX = saved_max
        CP._LINECACHE_KEYS.clear()
        CP._LINECACHE_KEYS.extend(saved_keys)


# ── K2: `execute_compiled` resolved the fncalls_compile verdict right after wrap,
# before the compiled callable's first REAL invocation ran -- poisoning it True even
# when the real invocation always fails. Fix: resolve only after that first call
# completes or raises. ──────────────────────────────────────────────────────────────

def test_k2_verdict_settles_false_when_the_first_real_call_fails(r: SubTestResult):
    """RED against the pre-fix ordering: a wrap that succeeds (`entry[1]` is a real
    backend name) but whose compiled callable raises on its very first real invocation
    must settle the fncalls_compile verdict FALSE, not TRUE. Constructed the same way
    B3#2 diagnosed it -- by reading the exact call ordering in `_compile_and_run`."""
    print("\n--- K2: a wrap-succeeds/first-call-fails fingerprint settles False, not True ---")
    prog, tm, used = _spatial_program()
    img = make_img(1, 16, 16, 3, seed=41)
    fp = "k2_test_fp"
    cache_key = (fp, "cpu", "fp32")
    C._compiled_cache.pop(cache_key, None)
    C._verify_state.pop(cache_key, None)
    C._route_memo.pop(fp, None)
    FC.reset_for_test()
    key = FC._key(fp, "cpu", "fp32")
    FC._pending.add(key)   # simulate _try_compile's own begin_attempt() having granted it

    def _raising_compiled_fn(*a, **kw):
        raise RuntimeError("simulated real Dynamo trace/lower failure at first call")

    orig_try_compile = C._try_compile
    C._try_compile = lambda *a, **kw: (_raising_compiled_fn, "inductor")
    try:
        C.execute_compiled(prog, {"A": img}, tm, "cpu", fp, output_names=["OUT"],
                           used_builtins=used)
        verdict = FC.verdict(fp, "cpu", "fp32")
        assert verdict is False, (
            f"expected the fncalls_compile verdict to settle False (the real "
            f"invocation raised), got {verdict!r} -- a wrap-time-only resolve would "
            f"have memoized True here, permanently, even though the artifact never "
            f"actually works")
        assert key not in FC._pending, "resolve_attempt must clear the pending marker"
        r.ok("K2: the verdict settles False from the real invocation's failure, not "
             "True from the wrap's own success")
    except Exception as e:
        r.fail("K2 verdict settles after real invocation", f"{type(e).__name__}: {e}")
    finally:
        C._try_compile = orig_try_compile
        C._compiled_cache.pop(cache_key, None)
        C._verify_state.pop(cache_key, None)
        C._route_memo.pop(fp, None)
        FC.reset_for_test()


def test_k2_verdict_settles_true_only_after_a_successful_real_call(r: SubTestResult):
    """GREEN companion: when the first real invocation actually succeeds, the verdict
    still settles True (K2 must not turn every compile permanently False)."""
    print("\n--- K2: a wrap-succeeds/first-call-succeeds fingerprint settles True ---")
    prog, tm, used = _spatial_program()
    img = make_img(1, 16, 16, 3, seed=42)
    fp = "k2_test_fp_ok"
    cache_key = (fp, "cpu", "fp32")
    C._compiled_cache.pop(cache_key, None)
    C._verify_state.pop(cache_key, None)
    C._route_memo.pop(fp, None)
    FC.reset_for_test()
    FC._pending.add(FC._key(fp, "cpu", "fp32"))

    calls = {"n": 0}

    def _ok_compiled_fn(program, bindings, type_map, device, latent_channel_count=0,
                        output_names=None, scale=None):
        calls["n"] += 1
        return bindings["A"]

    orig_try_compile = C._try_compile
    C._try_compile = lambda *a, **kw: (_ok_compiled_fn, "inductor")
    try:
        C.execute_compiled(prog, {"A": img}, tm, "cpu", fp, output_names=["OUT"],
                           used_builtins=used)
        assert calls["n"] == 1, f"expected the compiled callable invoked once, got {calls['n']}"
        verdict = FC.verdict(fp, "cpu", "fp32")
        assert verdict is True, f"expected the verdict to settle True, got {verdict!r}"
        r.ok("K2: a genuinely working artifact still settles True after its real call")
    except Exception as e:
        r.fail("K2 verdict settles true on success", f"{type(e).__name__}: {e}")
    finally:
        C._try_compile = orig_try_compile
        C._compiled_cache.pop(cache_key, None)
        C._verify_state.pop(cache_key, None)
        C._route_memo.pop(fp, None)
        FC.reset_for_test()


# ── K3: fncalls_compile's memo was keyed by bare fingerprint alone -- but whether a real
# compile succeeds is device/precision-dependent (a CPU-inductor-needs-MSVC-always-fails
# fingerprint says nothing about the SAME program on CUDA). Fix: widen the key to
# (fingerprint, device_type, precision), mirroring compiled.py's own cache_key. ──────────

def test_k3_memo_key_distinguishes_device_and_precision(r: SubTestResult):
    """RED against the pre-K3 bare-fingerprint key: a fingerprint resolved False on one
    (device, precision) must NOT settle the verdict for the SAME fingerprint on a
    different (device, precision) -- each combination gets its own real attempt."""
    print("\n--- K3: the memo key distinguishes device_type/precision, not fingerprint alone ---")
    fp = "k3_test_fp"
    FC.reset_for_test()
    try:
        assert FC.begin_attempt(fp, "cpu", "fp32") is True, (
            "the first attempt for (fp, cpu, fp32) must be granted")
        FC.resolve_attempt(fp, "cpu", "fp32", None)   # settles (fp, cpu, fp32) -> False
        assert FC.verdict(fp, "cpu", "fp32") is False
        # A DIFFERENT device for the SAME fingerprint must be a fresh, unresolved key --
        # not silently inheriting the cpu/fp32 verdict.
        assert FC.verdict(fp, "cuda", "fp32") is None, (
            "a (fp, cuda, fp32) query must not inherit (fp, cpu, fp32)'s settled verdict")
        assert FC.begin_attempt(fp, "cuda", "fp32") is True, (
            "(fp, cuda, fp32) must still be grantable -- it is a distinct key")
        FC.resolve_attempt(fp, "cuda", "fp32", "inductor")   # settles True
        assert FC.verdict(fp, "cuda", "fp32") is True
        assert FC.verdict(fp, "cpu", "fp32") is False, (
            "resolving the cuda/fp32 key must not disturb the already-settled cpu/fp32 one")
        # Same device, different precision: also a distinct key.
        assert FC.verdict(fp, "cpu", "fp16") is None
        r.ok("K3: (fingerprint, device_type, precision) are three independently "
             "resolvable keys for the same fingerprint")
    except Exception as e:
        r.fail("K3 memo key widened", f"{type(e).__name__}: {e}")
    finally:
        FC.reset_for_test()


def test_k3_persistence_round_trips_the_composite_key(r: SubTestResult):
    """The composite key must still round-trip through `snapshot_items()`/
    `adopt_persisted()` (what `warm_state.py` reads/writes) as a flat string -> bool
    map -- K3 must not turn this into a tuple that breaks JSON persistence."""
    print("\n--- K3: the composite key persists as a flat string, not a tuple ---")
    fp = "k3_persist_fp"
    FC.reset_for_test()
    try:
        FC.begin_attempt(fp, "cpu", "fp32")
        FC.resolve_attempt(fp, "cpu", "fp32", "inductor")
        items = FC.snapshot_items()
        assert all(isinstance(k, str) for k in items), (
            f"snapshot_items() must yield string keys (JSON-safe), got {list(items)!r}")
        key = FC._key(fp, "cpu", "fp32")
        assert items.get(key) is True
        FC.reset_for_test()
        FC.adopt_persisted(key, True)
        assert FC.verdict(fp, "cpu", "fp32") is True, (
            "adopt_persisted() must restore the verdict under the same composite key "
            "snapshot_items() emitted")
        r.ok("K3: the composite key round-trips as a flat, JSON-safe string")
    except Exception as e:
        r.fail("K3 composite key persistence", f"{type(e).__name__}: {e}")
    finally:
        FC.reset_for_test()


# ── K4: `begin_attempt`'s check-then-add across `_memo`/`_pending` was two separate
# statements with no lock -- two pools racing the SAME key could both observe "not yet
# pending" and both proceed. Fix: hold a lock across the check-and-set. ────────────────

def test_k4_begin_attempt_is_atomic_under_concurrent_callers(r: SubTestResult):
    """Forces the exact interleaving B3#3 describes: thread A's check (`key in _pending`)
    is made deliberately slow (standing in for a real thread switch mid-check, the same
    technique `test_phasec_fixcompile.py`'s C7 row uses for `_ES_CO_MEMO`), and thread B
    starts while A is still inside its own check. RED against the pre-K4 unlocked
    check-then-add: both threads observe 'not yet pending' and both get True. GREEN once
    the check-and-set is atomic: exactly one does."""
    print("\n--- K4: begin_attempt's check-then-add is atomic across concurrent callers ---")
    fp = "k4_race_fp"
    FC.reset_for_test()

    class _SlowContainsSet(set):
        """A plain set's `__contains__` can't be monkeypatched (it's a read-only slot on
        the builtin type), so force the race deterministically with a SUBCLASS instead,
        mirroring test_phasec_fixcompile.py's C7 `_SlowGetDict` technique exactly."""

        def __contains__(self, item):
            result = set.__contains__(self, item)
            time.sleep(0.05)   # stands in for a real thread switch mid-check
            return result

    orig_pending = FC._pending
    FC._pending = _SlowContainsSet()
    results = []
    lock = threading.Lock()

    def worker():
        got = FC.begin_attempt(fp, "cpu", "fp32")
        with lock:
            results.append(got)

    try:
        t1 = threading.Thread(target=worker)
        t2 = threading.Thread(target=worker)
        t1.start()
        time.sleep(0.01)   # ensure t1 is inside its own (slow) __contains__ check first
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)
        assert len(results) == 2, f"expected 2 results, got {len(results)}"
        assert sorted(results) == [False, True], (
            f"expected exactly ONE grant and one refusal, got {results!r} -- both threads "
            f"observed 'not yet pending' and both were granted the fall-through")
        r.ok("K4: exactly one of two concurrent callers is granted the fall-through "
             f"attempt for the same key ({results!r})")
    except Exception as e:
        r.fail("K4 begin_attempt atomicity", f"{type(e).__name__}: {e}")
    finally:
        FC._pending = orig_pending
        FC.reset_for_test()


# ── K5: a genuinely stuck job on `_COMPILE_POOL`/`_WARM_POOL` (max_workers=1, no
# timeout) blocked every LATER submission to that pool forever. Fix: a submission that
# finds the pool's current job older than `_POOL_STUCK_BOUND_S` abandons that pool for a
# fresh one, rather than queuing behind the stuck job forever. ─────────────────────────

def test_k5_a_stale_pool_is_replaced_not_queued_behind(r: SubTestResult):
    """RED against the pre-K5 shape (no staleness tracking at all -- a second submission
    to a busy pool just queues behind the first, however long the first takes): mark a
    pool "busy" long enough ago to exceed the bound, then confirm `_pool_for` swaps in a
    fresh pool object instead of returning the existing (stuck) one."""
    print("\n--- K5: a pool whose current job exceeds the stuck bound is replaced ---")
    saved_compile_pool = C._COMPILE_POOL
    saved_busy = dict(C._pool_busy_since)
    saved_bound = C._POOL_STUCK_BOUND_S
    try:
        C._POOL_STUCK_BOUND_S = 0.05
        stuck_pool = C._COMPILE_POOL
        C._mark_pool_busy("compile")
        # Simulate the bound having elapsed without a real 50ms sleep in the test.
        C._pool_busy_since["compile"] = C._time.monotonic() - 1.0
        fresh = C._pool_for("compile")
        assert fresh is not stuck_pool, (
            "a pool whose current job has run past the stuck bound must be replaced, "
            "not handed back unchanged")
        assert "compile" not in C._pool_busy_since, (
            "replacing a stale pool must also clear its busy marker")
        assert C._pool_for("compile") is fresh, (
            "a freshly-replaced, not-yet-busy pool must be returned unchanged on the "
            "very next call")
        r.ok("K5: a stale pool is abandoned for a fresh one; a fresh pool is not "
             "replaced again until it, too, goes stale")
    except Exception as e:
        r.fail("K5 stale pool replacement", f"{type(e).__name__}: {e}")
    finally:
        C._COMPILE_POOL = saved_compile_pool
        C._pool_busy_since.clear()
        C._pool_busy_since.update(saved_busy)
        C._POOL_STUCK_BOUND_S = saved_bound


def test_k5_busy_then_free_clears_the_marker_for_a_fast_job(r: SubTestResult):
    """GREEN companion: the ordinary (fast, not stuck) case -- busy then free within the
    bound must leave no stale marker behind for the NEXT submission to misread."""
    print("\n--- K5: an ordinary fast job's busy marker clears, no false staleness ---")
    saved_busy = dict(C._pool_busy_since)
    try:
        C._mark_pool_busy("warm")
        assert "warm" in C._pool_busy_since
        C._mark_pool_free("warm")
        assert "warm" not in C._pool_busy_since, (
            "a completed job must clear its own busy marker")
        r.ok("K5: busy/free bracket a job cleanly; no marker survives a normal completion")
    except Exception as e:
        r.fail("K5 busy/free bracket", f"{type(e).__name__}: {e}")
    finally:
        C._pool_busy_since.clear()
        C._pool_busy_since.update(saved_busy)
