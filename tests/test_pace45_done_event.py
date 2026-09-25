"""
PACE-45(a) — `CookResult.done: torch.cuda.Event | None`, a "GPU work done" fence recorded
on the cook's stream after its LAST launch, `None` off CUDA. Additive: no synchronize, and
costs nothing unless a caller reads/synchronizes it (see `tex_runtime/pacing.py::cook_done_event`,
wired in at `tex_engine.run`'s `CookResult(...)` construction).

The one CUDA-only row here (`test_pace45_done_event_cuda`) is a SIMP-3 skip site: there is
no CPU witness for a `torch.cuda.Event` at all. See `tests/test_simp3_skip_budget.py`'s
re-pin note. `tests/test_pace45_pacing.py` covers this ask's other half, paced cancellation.
"""
from helpers import *  # noqa: F401,F403  (SubTestResult, torch, make_img)
from TEX_Wrangle import tex_engine


def test_pace45_done_event_cpu(r: SubTestResult):
    """CookResult.done: None off CUDA."""
    res = tex_engine.cook("@OUT = vec4(@A.rgb, 1.0);", {"A": make_img(1, 8, 8, 4, seed=452)},
                          device_mode="cpu")
    if res.done is None:
        r.ok("CookResult.done is None on a CPU cook")
    else:
        r.fail("PACE-45 done event (CPU)", f"expected None, got {res.done!r}")


def test_pace45_done_event_cuda(r: SubTestResult):
    """CookResult.done: a CUDA event recorded after the cook's last launch, synchronizable,
    and additive -- present without anyone ever reading it (nothing in this cook's own
    output changes as a result)."""
    if not torch.cuda.is_available():
        r.skip("PACE-45 done event (CUDA)", "no CUDA on this box")
        return
    res = tex_engine.cook("@OUT = vec4(@A.rgb, 1.0);", {"A": make_img(1, 8, 8, 4, seed=453).cuda()},
                          device_mode="cuda")
    if not isinstance(res.done, torch.cuda.Event):
        r.fail("PACE-45 done event (CUDA)", f"expected a torch.cuda.Event, got {res.done!r}")
        return
    try:
        res.done.synchronize()
    except Exception as e:
        r.fail("PACE-45 done event (CUDA)", f".synchronize() raised: {e}")
        return
    r.ok("CookResult.done is a torch.cuda.Event on CUDA, and .synchronize() returns cleanly")
