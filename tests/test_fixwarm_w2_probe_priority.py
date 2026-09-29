"""
FIX-WARM W2 (v0.48 Phase C, B3#2) -- the capability probe must not wait behind a whole
`prewarm_async()` batch.

`prewarm_async()` (AUTO-48) submits `prewarm()` as ONE callable to the shared
`_DaemonProbePool` (`tex_runtime/compiled_capability.py`) -- the SAME single-worker,
strict-FIFO pool `compile_capability_async()` uses (AUTO-47). `prewarm()` processes its
whole `programs` list in one call before returning, so the ONE pool worker is held for the
ENTIRE batch, however many programs a host passes -- a project-load `prewarm_async` call
(the "many programs" case this ask targets) can starve the capability probe for the whole
warm-ahead duration, not "one cook's worth" as the sharing was designed to cost.

Red at 5ae6288 (B3's own structural repro, deterministic via a block/release rendezvous --
no wall-clock race): a THREE-program `prewarm_async` job with the middle program's own warm
step blocked; a `compile_capability_async()` probe submitted once the FIRST program has
already completed (so the probe is queued strictly behind only the STILL-PENDING remainder
of the batch, never behind work already done) must not have to wait for the batch at all.
The shipped fix gives `prewarm_async` its own pool (`_get_prewarm_pool()`), so the probe
resolves on the capability pool while the prewarm worker is still blocked inside program 1.
"""
import threading

import pytest

from helpers import *  # noqa: F401,F403  (TEXType, cold_engine_state)
from TEX_Wrangle import tex_api
from TEX_Wrangle.tex_runtime import compiled as C
from TEX_Wrangle.tex_runtime import compiled_capability as CC


def _three_programs():
    out = []
    for i in range(3):
        code = f"vec3 c=@A.rgb*1.1 - 0.{i}1; c=clamp(c,0.0,1.0); @OUT=vec4(c,1.0);"
        out.append((code, {"A": TEXType.VEC3}))
    return out


def test_fixwarm_w2_probe_not_blocked_by_whole_batch(r: SubTestResult):
    print("\n--- FIX-WARM W2: capability probe must not queue behind a WHOLE prewarm batch ---")
    with cold_engine_state():
        CC._reset_capability_cache_for_test()

        programs = _three_programs()
        program0_done = threading.Event()
        program1_gate = threading.Event()   # release to let program 1 (index 1) proceed
        program1_entered = threading.Event()
        seen = {"count": 0}
        orig = C._get_or_make_codegen_fn

        def gated(*a, **kw):
            i = seen["count"]
            seen["count"] += 1
            if i == 0:
                r_ = orig(*a, **kw)
                program0_done.set()
                return r_
            if i == 1:
                program1_entered.set()
                program1_gate.wait(timeout=10)
                return orig(*a, **kw)
            return orig(*a, **kw)

        C._get_or_make_codegen_fn = gated
        try:
            handle = tex_api.prewarm_async(programs, device="cpu", precision="fp32",
                                           compile_mode="auto")

            # Wait until program 0 has finished and program 1 is BLOCKED mid-emission --
            # the probe below is submitted strictly after the first program's work is
            # already done, so a probe that still has to wait for program 1 AND program 2
            # is being starved by the *remaining* batch, not by work already finished.
            assert program1_entered.wait(timeout=10), "program 1 never started"
            assert program0_done.is_set(), "program 0 never finished"

            probe_future = CC._get_capability_pool().submit(lambda: "PROBE_RAN")

            # Wait for the probe while program 1 is STILL gated (the gate is released only
            # below). A probe queued behind the batch cannot resolve until then, so it times
            # out here; one on its own pool resolves at once. No sleep, no early-done check.
            probe_result = None
            try:
                probe_result = probe_future.result(timeout=10)
            except Exception:
                pass
            probe_resolved_while_blocked = probe_result == "PROBE_RAN"

            program1_gate.set()   # let the rest of the batch finish
            summary = handle.wait(timeout=30)
        finally:
            C._get_or_make_codegen_fn = orig

        try:
            assert probe_resolved_while_blocked, (
                "the capability probe did not resolve while program 1 (2 of 3) was still "
                "blocked in the prewarm worker -- it is queued behind the prewarm batch "
                "instead of running on its own pool")
            assert summary["programs"] == 3 and summary["errors"] == 0, summary
            r.ok(f"probe resolved during the batch, not after it; summary={summary}")
        except AssertionError as e:
            r.fail("FIX-WARM W2 probe priority", str(e))
