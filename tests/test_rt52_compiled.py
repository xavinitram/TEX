"""Compiled-tier runtime: pool busy markers, MSVC discovery, limit classification, fn-calls verdicts, LRU touches."""
import sys
import threading
import uuid
from collections import OrderedDict

import torch

from helpers import *  # noqa: F401,F403
from TEX_Wrangle.tex_runtime import compiled as C
from TEX_Wrangle.tex_runtime import compiled_exec_support as ES
from TEX_Wrangle.tex_runtime import fncalls_compile as FC
from TEX_Wrangle.tex_runtime import lru_util


def test_late_job_of_an_abandoned_pool_keeps_the_replacement_marker():
    saved_pool, saved_busy, saved_bound = C._COMPILE_POOL, dict(C._pool_busy_since), C._POOL_STUCK_BOUND_S
    try:
        C._pool_busy_since.clear()
        C._POOL_STUCK_BOUND_S = 0.05
        old_token = C._mark_pool_busy("compile")
        C._pool_busy_since["compile"] = old_token = C._time.monotonic() - 1.0
        fresh = C._pool_for("compile")           # the stuck pool is abandoned
        assert "compile" not in C._pool_busy_since
        new_token = C._mark_pool_busy("compile")  # the replacement pool's job
        C._mark_pool_free("compile", old_token)   # the orphan finally finishes
        assert C._pool_busy_since.get("compile") == new_token
        C._mark_pool_free("compile", new_token)
        assert "compile" not in C._pool_busy_since
        assert fresh is not saved_pool
    finally:
        C._COMPILE_POOL = saved_pool
        C._pool_busy_since.clear()
        C._pool_busy_since.update(saved_busy)
        C._POOL_STUCK_BOUND_S = saved_bound


def test_msvc_search_covers_64bit_professional_and_takes_the_newest(monkeypatch):
    import glob as _glob
    seen = []
    prof = r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat"
    older = r"C:\Program Files\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvarsall.bat"

    def fake_glob(pattern, recursive=False):
        seen.append(pattern)
        if r"Program Files\Microsoft" in pattern and "Professional" in pattern:
            return [older, prof]
        return []

    ran = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("INCLUDE", raising=False)
    monkeypatch.setattr(_glob, "glob", fake_glob)
    monkeypatch.setattr(C.subprocess, "run",
                        lambda cmd, **kw: ran.append(cmd) or (_ for _ in ()).throw(RuntimeError("stop")))
    C._do_setup_msvc_env()
    assert ran and ran[0][3] == prof, ran


def test_user_limit_is_recognised_by_code_or_limit_phrase_not_the_word_iteration():
    assert ES._is_user_limit(RuntimeError("While loop exceeded maximum iteration limit (1024). Check"))
    assert ES._is_user_limit(RuntimeError("For loop would exceed 1024 iterations"))
    assert ES._is_user_limit(RuntimeError("Maximum function call depth exceeded in f()"))
    coded = RuntimeError("anything")
    coded.code = "E6010"
    assert ES._is_user_limit(coded)
    assert not ES._is_user_limit(RuntimeError("dictionary changed size during iteration"))
    assert not ES._is_user_limit(RuntimeError("Inductor lowering failed at iteration 3"))


def test_a_transient_or_environmental_failure_leaves_the_fncalls_verdict_unset():
    FC.reset_for_test()
    fp = "rt52_fnc_" + uuid.uuid4().hex     # a verdict recorded by an earlier run must not be adopted
    try:
        for exc in (MemoryError("x"), RuntimeError("CUDA out of memory"),
                    RuntimeError("Triton is not installed"), RuntimeError("Maximum function call depth exceeded")):
            assert FC.begin_attempt(fp, "cpu", "fp32")
            ES._settle_fncalls(fp, "cpu", "fp32", None, exc)
            assert FC.verdict(fp, "cpu", "fp32") is None
        assert FC.begin_attempt(fp, "cpu", "fp32")          # granted again after each
        ES._settle_fncalls(fp, "cpu", "fp32", None, RuntimeError("real dynamo failure"))
        assert FC.verdict(fp, "cpu", "fp32") is False
    finally:
        FC.reset_for_test()


def test_lru_get_touches_and_tolerates_a_vanished_key():
    d = OrderedDict([("a", 1), ("b", 2)])
    assert lru_util.lru_get(d, "a") == 1
    assert list(d) == ["b", "a"]
    assert lru_util.lru_get(d, "zzz", 7) == 7

    class Evicting(OrderedDict):
        def move_to_end(self, key, last=True):
            self.pop(key, None)                     # another thread evicted it just now
            raise KeyError(key)
    e = Evicting(a=1)
    assert lru_util.lru_get(e, "a", "gone") == "gone"


def test_a_cached_artifact_that_is_cooked_every_frame_moves_to_the_recent_end():
    hot, other = ("hot", "cpu", "fp32"), ("other", "cpu", "fp32")
    saved = OrderedDict(C._compiled_cache)
    try:
        C._compiled_cache.clear()
        fn = lambda program, b, tm, dev, lcc, on, scale=None: torch.zeros(1)
        C._compiled_cache[hot] = (fn, "inductor")
        C._compiled_cache[other] = (fn, "inductor")
        res, _ = C._run_cached_compiled(hot, None, {}, {}, "cpu", 0, None, "cpu", timed=False)
        assert res is not None
        assert list(C._compiled_cache)[-1] == hot
        C._compiled_cache.pop(hot)
        assert C._run_cached_compiled(hot, None, {}, {}, "cpu", 0, None, "cpu", timed=False) == (None, None)
    finally:
        C._compiled_cache.clear()
        C._compiled_cache.update(saved)


def _auto_setup():
    from TEX_Wrangle.tex_cache import parse_and_split
    from TEX_Wrangle.tex_compiler.type_checker import TypeChecker
    from TEX_Wrangle.tex_runtime.interpreter import _collect_identifiers
    code = "vec3 c=@A.rgb*1.3 - 0.1; c=clamp(c,0.0,1.0); @OUT=vec4(c,1.0);"
    bt = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}
    prog = parse_and_split(code, bt)
    tm = TypeChecker(binding_types=bt, source=code).check(prog)
    return prog, tm, _collect_identifiers(prog)


def test_a_finished_compile_is_trialled_even_after_an_idle_pause(tmp_path, monkeypatch):
    """The convergence bound counts wall-clock time, so a pause after the compile was submitted
    used to reject a healthy key (and persist the rejection). A finished compile is promoted
    first, and a bound-fired rejection is never written to disk."""
    from TEX_Wrangle.tex_runtime import autotier as AT
    prog, tm, used = _auto_setup()
    img = torch.rand(1, 8, 8, 3)
    fp = "rt52_idle_fp"
    cache_key = (fp, "cpu", "fp32")

    def fake_compiled(program, bindings, type_map, device, lcc, names, scale=None):
        return {n: torch.zeros(1, 8, 8, 4) for n in (names or ["OUT"])}

    monkeypatch.setattr(C, "_try_compile", lambda *a, **k: (fake_compiled, "inductor"))
    monkeypatch.setattr(C, "compile_capability_async",
                        lambda: {"cuda_inductor": True, "cpu_inductor": True, "reason": {}})
    AT.reset()
    try:
        for _ in range(3):
            C.run_auto(prog, {"A": img}, tm, "cpu", fp, output_names=["OUT"], used_builtins=used)
        C._drain_bg_for_test()
        C._bg_futures.pop(cache_key, None)
        key = AT.make_key(fp, "cpu", "fp32", C._consensus_extent({"A": img}, prog))
        assert AT.verdict(key) == AT.COMPILING and cache_key in C._compiled_cache
        AT._get(key).ready_wall -= AT._CONVERGENCE_BOUND_S + 5.0      # the user paused
        C.run_auto(prog, {"A": img}, tm, "cpu", fp, output_names=["OUT"], used_builtins=used)
        assert AT.verdict(key) != AT.REJECTED, "an idle pause rejected a compile that had finished"
    finally:
        C._drain_bg_for_test()
        C._compiled_cache.pop(cache_key, None)
        AT.reset()


def test_the_convergence_bound_is_not_persisted():
    from TEX_Wrangle.tex_runtime import autotier as AT
    AT.reset()
    key = ("rt52", "bound", "cpu", "fp32", 10, None)
    try:
        AT.record_interp(key, 5.0)
        for _ in range(3):
            AT.record_interp(key, 5.0)
        assert AT.should_submit_compile(key)
        AT.mark_submitted(key)
        AT._get(key).ready_wall -= AT._CONVERGENCE_BOUND_S + 1.0
        assert AT.enforce_convergence_bound(key)
        assert AT.verdict(key) == AT.REJECTED
        assert key in AT._NON_DURABLE
    finally:
        AT.reset()
