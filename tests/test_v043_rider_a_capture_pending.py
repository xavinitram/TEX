"""v0.43.0 rider (a) — `graphed.capture_pending(fingerprint, device) -> bool | None`.

A read-only peek at `graphed._capturable_memo`, the static AST capturability verdict
`run_graphed` memoizes per fingerprint (`docs/worklog/v043/design.md` §4(a)). Mechanism
only: no pre-trigger (never calls `_capturable`, never adopts a `warm_state`-persisted
verdict) and no side effect (never writes the memo).

DEVIATION from the rider's proposed signature, recorded here and in the hand-back:
`_capturable_memo` is `dict[str, tuple[bool, int]]` — keyed by `fingerprint` ALONE, because
graph capturability is a pure AST (+ arch) property with no device axis. `device` therefore
never indexes the memo; it only gates the CUDA-only precondition, exactly the way
`run_graphed` gates it (`dev.type != "cuda"`) BEFORE ever consulting the memo. So a
non-CUDA device is answered `False` (capture categorically cannot happen there — a known,
deterministic fact) without a memo lookup at all, never `None` ("unknown").

The three states this canaries, over the memo itself:
  * UNKNOWN     — fingerprint absent from `_capturable_memo` -> `None`.
  * CAPTURABLE  — memoized `(True, est_ops)` -> `True`.
  * NOT-CAPTURABLE — memoized `(False, est_ops)` -> `False`.
Plus the device gate (deterministic `False`, independent of the memo's content), and the
no-side-effect / no-pre-trigger guarantees.
"""
from helpers import *

from TEX_Wrangle.tex_runtime import graphed

_FP_UNKNOWN = "rt_a_fp_unknown_0001"
_FP_CAPTURABLE = "rt_a_fp_capturable_0002"
_FP_NOT_CAPTURABLE = "rt_a_fp_not_capturable_0003"


def _clean_memo():
    for fp in (_FP_UNKNOWN, _FP_CAPTURABLE, _FP_NOT_CAPTURABLE):
        graphed._capturable_memo.pop(fp, None)


def test_rt_a_capture_pending_three_memo_states(r: SubTestResult):
    print("\n--- RT-a: capture_pending over the memo's three states ---")
    _clean_memo()
    try:
        # UNKNOWN: nothing memoized yet for this fingerprint.
        got = graphed.capture_pending(_FP_UNKNOWN, "cuda")
        if got is not None:
            r.fail("RT-a unknown", f"expected None (unmemoized), got {got!r}")
            return

        # CAPTURABLE: a prior cook's AST gate memoized True.
        graphed._capturable_memo[_FP_CAPTURABLE] = (True, 3)
        got = graphed.capture_pending(_FP_CAPTURABLE, "cuda")
        if got is not True:
            r.fail("RT-a capturable", f"expected True, got {got!r}")
            return

        # NOT-CAPTURABLE: a prior cook's AST gate memoized False.
        graphed._capturable_memo[_FP_NOT_CAPTURABLE] = (False, 0)
        got = graphed.capture_pending(_FP_NOT_CAPTURABLE, "cuda")
        if got is not False:
            r.fail("RT-a not-capturable", f"expected False, got {got!r}")
            return

        r.ok("unknown -> None, capturable -> True, not-capturable -> False")
    finally:
        _clean_memo()


def test_rt_a_capture_pending_device_gate_is_deterministic(r: SubTestResult):
    print("\n--- RT-a: non-CUDA device answers False without consulting the memo ---")
    _clean_memo()
    try:
        # Memoize CAPTURABLE for this fingerprint, then ask about a CPU device: the
        # device gate must short-circuit to False, mirroring run_graphed's own
        # `dev.type != "cuda"` early return, and must NOT read the (capturable) memo
        # entry back as True — capture is categorically CUDA-only.
        graphed._capturable_memo[_FP_CAPTURABLE] = (True, 5)
        got = graphed.capture_pending(_FP_CAPTURABLE, "cpu")
        if got is not False:
            r.fail("RT-a device gate", f"expected False off CUDA, got {got!r}")
            return
        r.ok("cpu device -> False regardless of the memo's content")
    finally:
        _clean_memo()


def test_rt_a_capture_pending_no_side_effect(r: SubTestResult):
    print("\n--- RT-a: the peek never triggers the AST walk and never writes the memo ---")
    _clean_memo()
    try:
        before = dict(graphed._capturable_memo)
        graphed.capture_pending(_FP_UNKNOWN, "cuda")
        after = dict(graphed._capturable_memo)
        if after != before:
            r.fail("RT-a no side effect",
                   f"_capturable_memo changed: {before!r} -> {after!r}")
            return
        if _FP_UNKNOWN in after:
            r.fail("RT-a no side effect", "peek minted a memo entry for an unknown fingerprint")
            return
        r.ok("_capturable_memo untouched by an unmemoized peek")
    finally:
        _clean_memo()
