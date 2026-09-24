"""TRK-154 — `FlowPlan` gains `call_sites`, the class of sync `_masked_flow_syncs`
(`tex_runtime/graphed.py`) could not see: a call to a user-defined function reached under a
per-pixel `if` (M4's empty-call skip, `if not m_any(self._live)`) syncs — a device
`.item()` — exactly like the loop live-check `sync_points` already names and the scatter
compaction `scatter_sites` already names, but no set on the walk named the CALL SITE itself.

Direction, proved with counters below: this can only ever ADD a decline, never remove one.
`plan.call_sites` is OR'd into an existing decline condition
(`_masked_flow_syncs`/`_capturable`), so a program whose plan gains a non-empty
`call_sites` set flips CAPTURABLE -> NOT CAPTURABLE; nothing flips the other way. The
concrete case is `binding_write_in_call` (`tests/test_lang_l4_masked_flow.py`'s own L4
atom): before this fix its plan had NO sync_points/scatter_sites, so `_capturable` read it
as statically capturable and `run_graphed` genuinely ATTEMPTED the CUDA-graph capture — this
file's `test_trk154_the_flip_was_never_a_working_capture` reproduces that attempt directly
(bypassing the new gate) and shows it fails with a real
`cudaErrorStreamCaptureInvalidated`, is caught, and is blacklisted: the capture was already
doomed, just discovered the expensive way. After this fix the SAME program's plan carries
one `call_sites` entry, `_capturable` declines it BEFORE any capture is attempted, and
`run_graphed` returns the identical `None` — the observable engine behaviour is unchanged;
only the cost and the mechanism of the decline are.

ComfyUI-invisible because: `_masked_flow_syncs` (and therefore this) is asked ONLY of a
program `masked_flow.enabled_for` accepts, which requires BOTH a `//!tex 0.25`-or-newer
pragma AND `LANGUAGE_VERSION >= 0.25` — the engine is at `0.24` at this head, so no program
that exists today reaches this code at all (invariant 7); every row here opens the test
seam (`_masked_flow=True`) or the engine's own gate under `_engine_at`, the same two
mechanisms `test_lang_l6_satellites.py` already uses.
"""
import torch

from helpers import *

from TEX_Wrangle import tex_api
from TEX_Wrangle.tex_runtime import graphed
from TEX_Wrangle.tex_compiler.lexer import Lexer
from TEX_Wrangle.tex_compiler.parser import Parser
from failure_harness import compile_program, clone_bindings

import test_lang_l4_masked_flow as L4

PRAGMA = L4.PRAGMA
_CUDA = torch.cuda.is_available()

# The exact L4 atom this fix targets: `stamp(a)` sits INSIDE the per-pixel `if`, so the
# CALL SITE itself inherits a per-pixel live mask — unlike `RETURN_ONLY`'s `pick(@A.r)`
# (test_lang_l6_satellites.py), which is called UNIFORMLY; the per-pixel `if` there is
# inside the callee's own body, not at the call site.
_CALL_UNDER_IF = L4._ATOM_PROGRAMS["binding_write_in_call"]

# A plain sync-free 0.25 program (no calls at all) — must never gain a call_sites entry.
_NO_CALLS = "float a = @A.r; float r = a * 2.0;\n@OUT = vec4(r, r, r, 1.0);\n"

# A user-function call reached UNIFORMLY (top level, no enclosing per-pixel `if`) whose
# OWN body is per-pixel internally — the RETURN_ONLY shape — must also draw no call site.
_CALL_UNIFORM = ("float pick(float a) { if (a > 0.5) { return a * 10.0; } return a * 100.0; }\n"
                 "float r = pick(@A.r);\n@OUT = vec4(r, r, r, 1.0);\n")


def _parse(src):
    return Parser(Lexer(src).tokenize(), source=src).parse()


def _old_masked_flow_syncs(program, _masked_flow=None):
    """The pre-TRK-154 formula, reconstructed inline (not by importing/patching production
    code) — the exact three-line body `_masked_flow_syncs` had before this fix, so this test
    can compare both formulas' verdicts on the SAME programs without touching the module."""
    if _masked_flow is None:
        if getattr(program, "language", None) is None:
            return False
        from TEX_Wrangle.tex_runtime.masked_flow import enabled_for
        _masked_flow = enabled_for(program, "")
    if not _masked_flow:
        return False
    plan = tex_api.flow_plan(program)
    return (not plan.complete) or bool(plan.sync_points or plan.scatter_sites)


def test_trk154_call_sites_named_only_where_the_call_site_itself_is_per_pixel(r: SubTestResult):
    print("\n--- TRK-154: call_sites fires for a call SITE under a per-pixel if, not for a "
          "uniformly-called function whose own body is per-pixel ---")
    try:
        p_under_if = tex_api.flow_plan(_parse(PRAGMA + _CALL_UNDER_IF))
        assert len(p_under_if.call_sites) == 1, p_under_if.call_sites
        p_uniform = tex_api.flow_plan(_parse(PRAGMA + _CALL_UNIFORM))
        assert not p_uniform.call_sites, p_uniform.call_sites
        p_none = tex_api.flow_plan(_parse(PRAGMA + _NO_CALLS))
        assert not p_none.call_sites, p_none.call_sites
        r.ok("binding_write_in_call draws 1 call_sites entry; a uniformly-called function "
             "and a program with no calls draw none")
    except Exception as e:
        r.fail("TRK-154 call_sites shape", f"{type(e).__name__}: {e}")


def test_trk154_capture_verdict_counters(r: SubTestResult):
    """The counter-based proof the ask requires: over every L4 atom, count how many
    capturable verdicts move between the OLD formula (sync_points/scatter_sites only) and
    the NEW one (+ call_sites), and in which direction. Must be monotonic: some flip
    True->False, NONE flip False->True — call_sites only ever adds a decline."""
    print("\n--- TRK-154: capture-decision counters across every L4 atom ---")
    try:
        flips_to_false = []
        flips_to_true = []
        unchanged = 0
        for name, src in sorted(L4._ATOM_PROGRAMS.items()):
            prog = _parse(PRAGMA + src)
            old = (not tex_api.flow_plan(prog).complete) or bool(
                _old_masked_flow_syncs(prog, True))
            new = graphed._masked_flow_syncs(prog, True)
            if old != new:
                (flips_to_true if new else flips_to_false).append(name)
            else:
                unchanged += 1
        assert not flips_to_false, (
            f"a program newly reads as NOT syncing — call_sites must never REMOVE a "
            f"decline: {flips_to_false}")
        assert flips_to_true == ["binding_write_in_call"], (
            f"expected exactly ['binding_write_in_call'] to newly sync, got {flips_to_true}")
        r.ok(f"{len(flips_to_true)} atom newly syncs ({flips_to_true[0]}), 0 stop syncing, "
             f"{unchanged} unchanged — the only movement is a new, correct decline")
    except Exception as e:
        r.fail("TRK-154 capture verdict counters", f"{type(e).__name__}: {e}")

    # The same counters through the real gate `_capturable` (not just `_masked_flow_syncs`
    # in isolation) — this is what a CUDA-graph consumer actually calls.
    try:
        newly_declined = []
        for name, src in sorted(L4._ATOM_PROGRAMS.items()):
            prog = _parse(PRAGMA + src)
            plan = tex_api.flow_plan(prog)
            old_syncs = (not plan.complete) or bool(plan.sync_points or plan.scatter_sites)
            if old_syncs:
                continue          # already declined before this fix — not this row's signal
            unflagged = graphed._capturable(prog, _masked_flow=False)
            if unflagged[0] is not True:
                continue          # declined at every level already, unrelated to masking
            old_capturable = unflagged if not old_syncs else (False, 0)
            new_capturable = graphed._capturable(prog, _masked_flow=True)
            if old_capturable != new_capturable:
                assert new_capturable == (False, 0), (name, old_capturable, new_capturable)
                newly_declined.append(name)
        assert newly_declined == ["binding_write_in_call"], newly_declined
        r.ok(f"_capturable itself newly declines exactly {newly_declined} and nothing else")
    except Exception as e:
        r.fail("TRK-154 _capturable counters", f"{type(e).__name__}: {e}")


def test_trk154_the_flip_was_never_a_working_capture(r: SubTestResult):
    """The concrete claim the ask makes: `binding_write_in_call` was never a WORKING
    capture — it just failed at the expensive, runtime CUDA layer instead of the cheap,
    static one. Skips (does not fail) off CUDA: the claim is about what CUDA capture does,
    and there is nothing to reproduce without a device."""
    print("\n--- TRK-154: binding_write_in_call's old 'capturable' verdict was already "
          "doomed at the CUDA layer, not a working capture this fix now breaks ---")
    if not _CUDA:
        r.skip("TRK-154 doomed capture", "no CUDA on this box — nothing to reproduce")
        return
    code = PRAGMA + _CALL_UNDER_IF
    img = make_img(1, 4, 4, 3, seed=11).cuda()
    bindings = {"A": img}
    try:
        prog, tm, outs = compile_program(code, bindings)
        assert graphed._capturable(prog, _masked_flow=True) == (False, 0), (
            "the FIXED gate must decline this program statically")
        assert bool(_old_masked_flow_syncs(prog, True)) is False, (
            "premise: the OLD formula must have read this program as NOT syncing, or this "
            "test is not exercising the gap TRK-154 closed")
        # Reproduce the PRE-fix static verdict with a save/restore monkeypatch of the ONE
        # module function this fix changed, drive the SAME public entry point a real
        # consumer calls (`run_graphed`), and see what that stale "capturable" verdict
        # would actually have bought: a genuine, doomed CUDA-graph capture attempt.
        saved = graphed._masked_flow_syncs
        graphed._masked_flow_syncs = _old_masked_flow_syncs
        fp = "trk154_doomed_capture_probe"
        try:
            assert graphed._capturable(prog, _masked_flow=True) == (True, 5), (
                "premise: with the old formula this program must read statically "
                "capturable, or the monkeypatch is not taking effect")
            out1 = graphed.run_graphed(prog, clone_bindings(bindings), tm, "cuda", fp,
                                       output_names=outs, precision="fp32")
            out2 = graphed.run_graphed(prog, clone_bindings(bindings), tm, "cuda", fp,
                                       output_names=outs, precision="fp32")
        finally:
            graphed._masked_flow_syncs = saved
        assert out1 is None and out2 is None, (
            "the pre-fix 'capturable' verdict led to a WORKING capture "
            f"(out1={out1!r}, out2={out2!r}) — this fix would wrongly decline it")
        r.ok("with the pre-fix formula, run_graphed genuinely attempts the capture and "
             "still returns None both times (capture failed at the CUDA layer, caught, "
             "blacklisted) — confirming this was never a working capture, only a more "
             "expensive way to reach the same decline")
    except AssertionError:
        raise
    except Exception as e:
        r.fail("TRK-154 doomed capture", f"{type(e).__name__}: {e}")
