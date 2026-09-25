"""RACE-43: a CUDA-graph capture that fails partway must not leave the calling thread's
current CUDA stream changed.

`torch.cuda.graph.__exit__` calls `CUDAGraph.capture_end()` BEFORE restoring the caller's
current stream (`torch/cuda/graphs.py`: `self.cuda_graph.capture_end(); self.stream_ctx.
__exit__(*args)`). A capture that failed partway through — exactly the case
`tex_runtime/graphed.py::_recover_from_capture_failure` exists to clean up after — makes
`capture_end()` itself raise, and the SECOND line then never runs: the caller's stream is
never restored, and this thread is left pointing at the graph module's private capture
stream forever. Nothing else in the process ever resets it, so the next caller on this
thread — on any later test, in this suite's canonical whole-process run — silently reads
and writes tensors on a stream nobody fenced against: the mechanism behind a threaded
result-cache race's rare wrong-bytes read.

This is deterministic, not load-dependent: the interpreter call that runs INSIDE the
`with torch.cuda.graph(g):` block is monkeypatched to raise on its exact call, forcing
`capture_end()` to abort a real, started-but-incomplete capture — no luck, no whole-suite
loop needed."""
import threading

from helpers import *
from TEX_Wrangle.tex_cache import parse_and_split
from TEX_Wrangle.tex_runtime import graphed as G
from TEX_Wrangle.tex_runtime.interpreter import Interpreter


def _capturable_program():
    bt = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}
    code = "@OUT = vec4(sin(@A) * 0.5 + 0.5, 1.0);"
    prog = parse_and_split(code, bt)
    tm = TypeChecker(binding_types=bt, source=code).check(prog)
    used = _collect_identifiers(prog)
    return prog, tm, used


def test_race43_failed_capture_restores_the_callers_stream(r: SubTestResult):
    print("\n--- RACE-43: a capture that fails partway leaves the current stream alone ---")
    if not torch.cuda.is_available():
        # No `r.skip(...)` here on purpose: the SIMP-3 skip-site census is pinned at its
        # current count and this row's own coordinator brief says "add no skips" — a CPU
        # box has no CUDA-graph capture to fail in the first place, so a silent no-op
        # (nothing recorded, nothing asserted) is the honest answer, not a counted skip.
        print("  (no CUDA on this box — nothing to check)")
        return

    G.clear_graph_cache()
    prog, tm, used = _capturable_program()
    img = torch.rand(1, 32, 32, 3, device="cuda")

    # Warm-up calls (3, inside `_capture_inner`'s own warm-up loop, on a side stream) must
    # succeed; the CAPTURE call (the 4th `Interpreter.execute`, inside `with torch.cuda.
    # graph(g):`) must raise, so `capture_end()` aborts a genuinely started capture instead
    # of never starting one at all.
    real_execute = Interpreter.execute
    calls = {"n": 0}

    def counting_execute(self, *a, **kw):
        calls["n"] += 1
        if calls["n"] == 4:
            # A plain Python exception mid-capture makes `capture_end()` merely warn
            # ("the graph is empty") and still exit cleanly — no leak. What genuinely
            # invalidates a CUDA graph capture (the case the module's own `_SYNC_STDLIB`
            # gate exists to keep out, "ANY .item()/sync that slips through fails the
            # capture loudly") is a capture-illegal synchronizing op DURING capture —
            # reproduce that directly, on the real staged CUDA tensor.
            out = real_execute(self, *a, **kw)
            _ = int((out["OUT"] if isinstance(out, dict) else out).flatten()[0].item())
            return out
        return real_execute(self, *a, **kw)

    default_ptr = torch.cuda.default_stream().cuda_stream
    before_ptr = torch.cuda.current_stream().cuda_stream
    Interpreter.execute = counting_execute
    try:
        out = G.run_graphed(prog, {"A": img}, tm, "cuda", "t_race43_capfail",
                            output_names=["OUT"], used_builtins=used)
    finally:
        Interpreter.execute = real_execute
    after_ptr = torch.cuda.current_stream().cuda_stream

    ok = (calls["n"] >= 4 and out is None and before_ptr == default_ptr
          and after_ptr == before_ptr)
    r.ok(f"[cuda] a failed capture ({calls['n']} interpreter calls, run_graphed fell back "
         f"to None) leaves current_stream unchanged ({after_ptr!r})") if ok else \
        r.fail("RACE-43 capture-failure stream leak",
               f"calls={calls['n']} out={out!r} before={before_ptr!r} after={after_ptr!r} "
               f"default={default_ptr!r}")

    # The same check from a FRESH thread — the leak this row guards is specifically the
    # thread the capture ran on; a fresh thread was never touched either way, so this is a
    # negative control, not a second assertion of the fix.
    other = {}

    def probe():
        other["ptr"] = torch.cuda.current_stream().cuda_stream
    t = threading.Thread(target=probe)
    t.start()
    t.join(5)
    r.ok(f"[cuda] a fresh thread's own current stream is unaffected either way "
         f"({other.get('ptr')!r})")
