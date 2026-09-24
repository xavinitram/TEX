"""
v0.42 HOSTAUDIT-2 — `tex_api.prewarm` now polls a `CancelToken` between programs.

Inventory (this ask): every long-running engine path was checked for a `CancelToken` poll.
`prewarm` iterates an arbitrary-length, host-supplied list of programs, and per program does
a full compile + codegen emission (+ a non-blocking background `torch.compile` submit) — the
one span in the inventory that was ENTIRELY un-polled, start to finish, with no yield point at
all (compare `tex_engine.run`'s yields A/C/D/E, or `tex_memory`'s per-strip yield F). Everything
else named in the ask either already polls (the strip/tile execution loops in `tex_memory.py`)
or is bounded by construction and does not need to (ROI/tile PLANNING and fusion detection are
capped at the 16-stage fusion limit; the compile/CUDA-graph tiers have no internal yield point
BY DESIGN — see `tex_engine._roi_codegen_exec`'s docstring — because a torch.compile call or a
CUDA-graph capture cannot be safely interrupted mid-call, only skipped before it starts).

Unlike a cook's cancel, a cancelled `prewarm` never raises `CookCancelled` — every program
already warmed keeps its (already-persisted) verdict, and warming is best-effort by contract
(the docstring already says "a bad program is skipped, never fatal"). The caller gets its usual
summary dict, with `summary["cancelled"]` counting how many programs were skipped.
"""
from TEX_Wrangle.tex_compiler.types import TEXType


def _progs(n):
    return [(f"@OUT = @A * {1.0 + i * 0.01};", {"A": TEXType.VEC4}) for i in range(n)]


class _CancelAfter:
    """A CancelToken that raises CookCancelled once `.check()` has been called `trips` times."""
    def __init__(self, trips: int):
        self.trips = trips
        self.calls = 0

    def check(self) -> None:
        from TEX_Wrangle.tex_runtime.host import CookCancelled
        self.calls += 1
        if self.calls > self.trips:
            raise CookCancelled("test: prewarm cancelled")


def test_hostaudit2_prewarm_honours_cancel_between_programs(r):
    from TEX_Wrangle import tex_api
    from TEX_Wrangle.tex_runtime import warm_state
    try:
        warm_state._reset_for_test()
        progs = _progs(5)
        token = _CancelAfter(trips=2)   # allow 2 programs through, then trip
        summary = tex_api.prewarm(progs, device="cpu", compile_mode="none", cancel=token)
        if summary["programs"] == 2 and summary["cancelled"] == 3:
            r.ok(f"prewarm stopped after 2 programs on cancel "
                 f"(programs={summary['programs']}, cancelled={summary['cancelled']})")
        else:
            r.fail("prewarm cancel grain",
                   f"expected programs=2, cancelled=3; got {summary}")
    except Exception as e:
        r.fail("prewarm cancel grain", f"{type(e).__name__}: {e}")


def test_hostaudit2_prewarm_never_raises_cookcancelled(r):
    """The prewarm contract is best-effort/never-fatal; a cancel must not become an exception
    a caller has to catch (that would be a breaking change to every existing caller)."""
    from TEX_Wrangle import tex_api
    from TEX_Wrangle.tex_runtime import warm_state
    from TEX_Wrangle.tex_runtime.host import CookCancelled
    try:
        warm_state._reset_for_test()
        progs = _progs(3)
        token = _CancelAfter(trips=0)   # trip on the very first check
        summary = tex_api.prewarm(progs, device="cpu", compile_mode="none", cancel=token)
        if summary["programs"] == 0 and summary["cancelled"] == 3:
            r.ok("prewarm cancelled before the first program still returns a summary, no raise")
        else:
            r.fail("prewarm immediate cancel", f"got {summary}")
    except CookCancelled:
        r.fail("prewarm never-raises contract",
               "prewarm raised CookCancelled — this breaks every caller that does not expect it")
    except Exception as e:
        r.fail("prewarm immediate cancel", f"{type(e).__name__}: {e}")


def test_hostaudit2_prewarm_without_cancel_is_unchanged(r):
    """`cancel` is optional and additive — every existing caller (none of which passes it)
    must warm every program exactly as before."""
    from TEX_Wrangle import tex_api
    from TEX_Wrangle.tex_runtime import warm_state
    try:
        warm_state._reset_for_test()
        progs = _progs(3)
        summary = tex_api.prewarm(progs, device="cpu", compile_mode="none")
        if summary["programs"] == 3 and summary["cancelled"] == 0:
            r.ok("prewarm with no cancel= argument still warms every program")
        else:
            r.fail("prewarm default-path regression", f"got {summary}")
    except Exception as e:
        r.fail("prewarm default-path regression", f"{type(e).__name__}: {e}")
