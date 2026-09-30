"""`torch._dynamo.reset()` must never run while another thread is inside Dynamo work.

Dynamo's caches are process-global. A reset frees the compiled entries, guards and resume
functions a running compile or compiled frame is still using, and the process dies in native
code: SIGFPE on Linux, an access violation on Windows. A whole-suite run died this way when a
slow compile outlived the example harness's per-program timeout and the harness reset Dynamo
from the main thread while the compile kept running on the compile pool. The product's own
failure paths reset the same way while the other compile pool, or an abandoned one, may still
be running a job.

The gate (`tex_runtime/dynamo_gate.py`) counts every pool job, and every reset TEX does goes
through `reset_if_idle()`, which skips while any job runs.

PORTABILITY: CPU only. No compiler toolchain needed: the compile is a stub, and the gate
needs no backend at all.
"""
import threading

import pytest
import torch

from helpers import *  # noqa: F401,F403
from TEX_Wrangle.tex_cache import parse_and_split
from TEX_Wrangle.tex_runtime import compiled as C
from TEX_Wrangle.tex_runtime import compiled_capability as CC
from TEX_Wrangle.tex_runtime import dynamo_gate as G


def _program():
    code = "vec3 c=@A.rgb; c = c*1.3 - 0.1; c = clamp(c, 0.0, 1.0); @OUT=vec4(c,1.0);"
    bt = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}
    prog = parse_and_split(code, bt)
    return prog, TypeChecker(binding_types=bt, source=code).check(prog)


@pytest.fixture
def resets(monkeypatch):
    calls = []
    monkeypatch.setattr(torch._dynamo, "reset", lambda: calls.append(1))
    return calls


@pytest.fixture
def failing_compile(monkeypatch):
    """`_try_compile` hands back a compiled callable whose first call raises, the shape a
    real first-call trace failure takes, so `execute_compiled` walks its failure path."""
    def _boom(*a, **k):
        raise RuntimeError("simulated first-call compile failure")
    monkeypatch.setattr(C, "_try_compile", lambda *a, **k: (_boom, "inductor"))
    monkeypatch.setattr(C, "_COMPILE_OP_THRESHOLD", 0)   # the tiny program takes the compile route


def _block(pool):
    """Occupy `pool` with a job that waits until the returned event is set."""
    started, release = threading.Event(), threading.Event()

    def job():
        started.set()
        release.wait(30)
    fut = pool.submit(job)
    assert started.wait(10), "the blocking job never started"
    return release, fut


def _cook(fp):
    prog, tm = _program()
    img = make_img(1, 8, 8, 3, seed=5)
    out = C.execute_compiled(prog, {"A": img}, tm, "cpu", fp, output_names=["OUT"])
    assert out["OUT"].shape == (1, 8, 8, 4)   # the interpreter fallback still answers


@pytest.mark.parametrize("pool_attr", ["_WARM_POOL", "_COMPILE_POOL_ABANDONED"])
def test_a_failed_compile_does_not_reset_dynamo_while_a_pool_job_runs(
        r, resets, failing_compile, pool_attr):
    # The warm pool runs concurrently with the compile pool, and an abandoned pool (one
    # `_pool_for` replaced) keeps running its stuck job: both are live Dynamo users.
    pool = (C._WARM_POOL if pool_attr == "_WARM_POOL"
            else CC._CompilePool("tex-gate-abandoned"))
    release, fut = _block(pool)
    try:
        _cook(f"gate_busy_{pool_attr}")
        assert not resets, "Dynamo was reset while another thread's job was still running"
    finally:
        release.set()
        fut.result(timeout=10)
    r.ok(f"no reset while a {pool_attr} job runs")


def test_a_failed_compile_still_resets_dynamo_when_nothing_runs(r, resets, failing_compile):
    assert G.active_jobs() == 0
    _cook("gate_idle")
    assert resets == [1], resets
    r.ok("the failure path still resets Dynamo when no job is running")


def test_reset_if_idle_is_atomic_with_job_entry(r, resets):
    with G.dynamo_job():
        assert G.reset_if_idle() is False
    assert G.reset_if_idle() is True
    assert resets == [1]
    r.ok("reset skipped inside a job, run once outside")

