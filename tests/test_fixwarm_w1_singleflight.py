"""
FIX-WARM W1 (v0.48 Phase C, B3#1) -- `_get_or_make_codegen_fn` single-flight emission.

`tex_runtime/compiled.py::_get_or_make_codegen_fn` is a bare check-then-act: it reads
`TEXCache.get_codegen_fn(fp)`, and on a miss calls `_try_codegen(...)` +
`store_codegen_fn(fp, cg_fn)` with no lock and no in-flight marker. `prewarm_async()`'s
entire point is to warm a program concurrently with a live cook of that SAME program (an
ordinary race: undo/redo, a re-queued node, a host that starts a background warm for the
next-likely program while the current one is still cooking) -- and when both callers land
on the identical fingerprint before either has stored a result, BOTH independently pay the
full emit (AST walk + codegen build + marshal + sha256 + disk write) for a cost this path
exists specifically to avoid paying twice.

Red at 5ae6288 (B3's own repro shape): two threads racing `_get_or_make_codegen_fn` for the
IDENTICAL fingerprint, with `_try_codegen` wrapped to count calls and widen the window with
a short sleep, produce 2 emissions. The fix makes the second caller for an in-flight
fingerprint wait on the first's result instead of redoing the work: 1 emission, both callers
get the same (or an equivalent) usable fn.
"""
import threading

import pytest

from helpers import *  # noqa: F401,F403  (TEXType, cold_engine_state)
from TEX_Wrangle.tex_runtime import compiled as C


def _one_program():
    code = "vec3 c = @A.rgb * 1.7 - 0.31; c = clamp(c, 0.0, 1.0); @OUT = vec4(c, 1.0);"
    bt = {"A": TEXType.VEC3}
    return code, bt


def test_fixwarm_w1_concurrent_same_fingerprint_emits_once(r: SubTestResult):
    print("\n--- FIX-WARM W1: concurrent _get_or_make_codegen_fn for one fingerprint ---")
    with cold_engine_state():
        from TEX_Wrangle import tex_api
        from TEX_Wrangle.tex_cache import get_cache
        code, bt = _one_program()
        fp = get_cache().fingerprint(code, bt)
        prog, fp = tex_api._compile_impl(code, bt, fp=fp)

        calls = []
        call_lock = threading.Lock()
        started = threading.Event()
        orig = C._try_codegen

        def counting_slow_codegen(*a, **kw):
            with call_lock:
                calls.append(1)
            # Signal AFTER registering the call but BEFORE the (slow, real) emission
            # finishes, so the second thread is guaranteed to reach
            # `_get_or_make_codegen_fn` while the first is genuinely still emitting —
            # deterministic overlap, not a scheduling race (B3's own widen-the-window
            # technique, without needing both threads to enter this wrapper).
            started.set()
            import time
            time.sleep(0.15)
            return orig(*a, **kw)

        C._try_codegen = counting_slow_codegen
        results = [None, None]
        errors = []

        def leader():
            try:
                results[0] = C._get_or_make_codegen_fn(prog.ast, prog.type_map, fp)
            except Exception as e:  # pragma: no cover -- surfaced via `errors`
                errors.append(e)

        def follower():
            try:
                assert started.wait(timeout=10), "leader never began emitting"
                results[1] = C._get_or_make_codegen_fn(prog.ast, prog.type_map, fp)
            except Exception as e:  # pragma: no cover -- surfaced via `errors`
                errors.append(e)

        try:
            t0 = threading.Thread(target=leader)
            t1 = threading.Thread(target=follower)
            t0.start()
            t1.start()
            t0.join(timeout=10)
            t1.join(timeout=10)
        finally:
            C._try_codegen = orig

        try:
            assert not errors, f"worker thread(s) raised: {errors}"
            assert len(calls) == 1, (
                f"expected exactly ONE codegen emission for one fingerprint raced by two "
                f"callers, got {len(calls)} -- _get_or_make_codegen_fn is not single-flight")
            assert results[0] is not None and results[1] is not None, (
                f"both callers must get a usable fn back, got {results}")
            r.ok(f"one fingerprint, two racing callers -> {len(calls)} emission(s)")
        except AssertionError as e:
            r.fail("FIX-WARM W1 single-flight", str(e))
