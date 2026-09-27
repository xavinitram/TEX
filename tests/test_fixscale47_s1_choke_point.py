"""FIX-SCALE S1 (v0.47 Phase C, B2 finding 1) — the scale-safety refusal is a SINGLE choke
point every scale-accepting entry point goes through, not only `tex_engine.prepare()`'s own
single-program path.

Before this fix: `tex_chain.cook_stage_list`, `tex_checkpoint.cook_checkpointed` and
`tex_checkpoint.materialize` all accepted `scale=` and threaded it straight to
`Interpreter.execute` without ever consulting `tex_roi.scale_verdict`/`scale_safe` — a program
the classifier itself declares unsafe (reads `ix`/`iy` outside a whitelisted `fetch` call)
cooked coarse through any of the three with no refusal at all, in direct contradiction of
AUTHOR DECISION #2 ("R3 classifier default = unsafe -> never coarse")."""
import torch

from helpers import *
from TEX_Wrangle import tex_chain, tex_checkpoint, tex_engine, tex_results

_UNSAFE_CODE = "@OUT = vec4(ix * 0.01, iy * 0.01, 0.0, 1.0);"


def _img():
    return torch.ones(1, 4, 4, 3)


def _two_stage_unsafe_tail(img):
    """A linear, collapsed two-stage chain whose TERMINAL stage is the classifier's own
    canonical unsafe program (hand pixel arithmetic outside a whitelisted fetch call)."""
    return [
        {"code": "@OUT = vec4(@A.rgb * 1.5, 1.0);", "chain_input": None, "bindings": {"A": img}},
        {"code": _UNSAFE_CODE, "chain_input": "X", "bindings": {}},
    ]


def _assert_refused(r: SubTestResult, label: str, fn):
    raised = None
    try:
        fn()
    except Exception as e:
        raised = e
    if raised is None:
        r.fail(label, "no exception was raised for an unsafe program at a non-trivial scale "
               "-- it cooked coarse silently")
        return False
    refusal = getattr(raised, "tex_refusal", None)
    if refusal is None or refusal.code != "scale-unsafe":
        r.fail(label, f"expected .tex_refusal.code == 'scale-unsafe', got {refusal!r} on "
               f"{type(raised).__name__}: {raised}")
        return False
    r.ok(f"{label}: refused with tex_refusal.code={refusal.code!r}")
    return True


def test_s1_cook_stage_list_refuses_unsafe_at_scale(r: SubTestResult):
    print("\n--- FIX-SCALE S1: cook_stage_list(scale=0.5) on an unsafe terminal stage refuses ---")
    stages = _two_stage_unsafe_tail(_img())
    _assert_refused(r, "cook_stage_list",
                    lambda: tex_chain.cook_stage_list(stages, device="cpu", scale=0.5))


def test_s1_cook_stage_list_single_stage_also_refuses(r: SubTestResult):
    print("\n--- FIX-SCALE S1: the single-stage (no fusion) path also refuses ---")
    stages = [{"code": _UNSAFE_CODE, "chain_input": None, "bindings": {}}]
    _assert_refused(r, "cook_stage_list (single stage)",
                    lambda: tex_chain.cook_stage_list(stages, device="cpu", scale=0.5))


def test_s1_cook_checkpointed_refuses_unsafe_at_scale(r: SubTestResult):
    print("\n--- FIX-SCALE S1: cook_checkpointed(scale=0.5) on an unsafe terminal stage refuses ---")
    stages = _two_stage_unsafe_tail(_img())
    # No ResultCache -> the CACHE-6 gate refuses -> `_full()`, which is exactly the route
    # B2's repro used: the documented, intended entry point for progressive refinement.
    _assert_refused(r, "cook_checkpointed (no cache, _full route)",
                    lambda: tex_checkpoint.cook_checkpointed(stages, None, device="cpu",
                                                             scale=0.5))


def test_s1_materialize_refuses_unsafe_at_scale(r: SubTestResult):
    print("\n--- FIX-SCALE S1: materialize(scale=0.5) on an unsafe terminal stage refuses ---")
    stages = _two_stage_unsafe_tail(_img())
    rc = tex_results.ResultCache()
    # An explicit single-edge cut with a real cache + a matching upstream key count admits
    # the CACHE-6 gate, so this drives the TAPPED cook_stage_list call, not the early-return
    # `if not cuts: return []` path that would cook nothing.
    _assert_refused(r, "materialize (tapped cook route)",
                    lambda: tex_checkpoint.materialize(stages, rc, cuts=[1],
                                                       upstream=("srckey",), device="cpu",
                                                       scale=0.5))


def test_s1_unsafe_program_still_cooks_at_scale_none_or_one(r: SubTestResult):
    print("\n--- FIX-SCALE S1: scale=None/1.0 never refuses through any of the three (invariant #7) ---")
    stages = _two_stage_unsafe_tail(_img())
    try:
        out_none = tex_chain.cook_stage_list(stages, device="cpu")
        out_one = tex_chain.cook_stage_list(stages, device="cpu", scale=1.0)
        rc = tex_results.ResultCache()
        tex_checkpoint.cook_checkpointed(stages, None, device="cpu")
        tex_checkpoint.materialize(stages, rc, cuts=[1], upstream=("srckey",), device="cpu")
    except Exception as e:
        r.fail("no spurious refusal", f"unexpected raise at scale=None/1.0: {e}")
        return
    if "OUT" not in out_none or "OUT" not in out_one:
        r.fail("cook shape", f"expected OUT in both: {out_none.keys()} / {out_one.keys()}")
        return
    r.ok("scale=None and scale=1.0 cook an 'unsafe' program normally through all three entries")
