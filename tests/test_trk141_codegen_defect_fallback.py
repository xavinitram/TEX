"""TRK-141 (LANG-L2 F1) — a bare `_CgBreak`/`_CgContinue` escaping a compile
tier's callable was swallowed by that tier's own `except Exception` exactly like
an ordinary decline (a missing backend, an unsupported construct): a
`logger.warning` with the bare `str(exc)` (empty for these two classes) and NO
`tier_trace.record` call at all. `tier_trace.last()` therefore still read the
PRIOR cook's record (or `None`, right after `prepare()`'s `tier_trace.reset()`),
indistinguishable from "no fallback happened" — the interpreter served its own
(possibly wrong) value with no diagnostic reaching the node, on every one of the
torch_compile / auto / cuda_graph strategies (`tex_engine.py`'s three
`except Exception as {compile_exc,auto_exc,_g_exc}:` sites, verified at head
`5c8cf7d` to sit at lines 451/467/483 exactly as the row cites them).

Fixed at those same three sites: `_record_codegen_defect_fallback` (added right
after `_interp_fallback`) special-cases ONLY `_CgBreak`/`_CgContinue` — logging
at ERROR with the class named and calling `tier_trace.record("interpreter",
fallback_from=<tier>, reason=...)` — so this ONE class of exception is never
silent. An ordinary decline (any other exception) is untouched: same warning,
still no tier_trace record, same fallback value — so this is additive on the
already-existing "never hard-fail the node" contract, not a change to it.

ComfyUI-invisible because: no TEX program a ComfyUI user can author raises
`_CgBreak`/`_CgContinue` at a tier boundary today (the one known way to reach it,
a `break`/`continue` leaving a function's own scope, is now refused at
type-check time as `E3015` — TRK-28); this row's own test drives it only by
monkeypatching the compiled-tier entry point to raise the class directly, which
is not a program a host can construct. The fix changes what gets LOGGED and
recorded in a diagnostics-only trace on an already-unreachable path; it moves no
pixel of any cook a ComfyUI user can produce.
"""
from helpers import *

from TEX_Wrangle import tex_engine
from TEX_Wrangle.tex_runtime import tier_trace

_CODE = "@OUT = @A * 1.5;"


def _cg_break_bindings():
    return {"A": make_img(1, 4, 4, 3, seed=141)}


def test_trk141_torch_compile_cgbreak_is_recorded_not_silent(r: SubTestResult):
    """`_run_torch_compile`'s catch: a `_CgBreak` escaping `execute_compiled` must
    log at ERROR and leave a tier_trace record naming the class, not nothing."""
    print("\n--- TRK-141: torch_compile fallback on a bare _CgBreak is recorded ---")
    bindings = _cg_break_bindings()
    orig = tex_engine.execute_compiled

    def _raise_cgbreak(*a, **kw):
        raise _CgBreak()

    tex_engine.execute_compiled = _raise_cgbreak
    try:
        tier_trace.reset()
        result = tex_engine.cook(_CODE, dict(bindings), device_mode="cpu",
                                 compile_mode="torch_compile")
        rec = tier_trace.last()
        if result is None or "OUT" not in result.outputs:
            r.fail("torch_compile still serves a result on the codegen-defect path",
                  "cook() returned no OUT")
        elif rec is None:
            r.fail("the fallback is recorded in tier_trace", "tier_trace.last() is still None")
        elif rec.tier != "interpreter" or rec.fallback_from != "torch_compile":
            r.fail("the record names the interpreter + torch_compile fallback",
                  repr(rec))
        elif "_CgBreak" not in (rec.reason or ""):
            r.fail("the record's reason names the _CgBreak class", repr(rec))
        else:
            r.ok(f"recorded and non-silent: {rec!r}")
    finally:
        tex_engine.execute_compiled = orig


def test_trk141_auto_cgcontinue_is_recorded_not_silent(r: SubTestResult):
    """`_run_auto`'s catch: a `_CgContinue` escaping `run_auto` must log at ERROR
    and leave a tier_trace record naming the class, not nothing."""
    print("\n--- TRK-141: auto fallback on a bare _CgContinue is recorded ---")
    bindings = _cg_break_bindings()

    import TEX_Wrangle.tex_runtime.compiled as compiled_mod
    orig = compiled_mod.run_auto

    def _raise_cgcontinue(*a, **kw):
        raise _CgContinue()

    compiled_mod.run_auto = _raise_cgcontinue
    try:
        tier_trace.reset()
        result = tex_engine.cook(_CODE, dict(bindings), device_mode="cpu",
                                 compile_mode="auto")
        rec = tier_trace.last()
        if result is None or "OUT" not in result.outputs:
            r.fail("auto still serves a result on the codegen-defect path",
                  "cook() returned no OUT")
        elif rec is None:
            r.fail("the fallback is recorded in tier_trace", "tier_trace.last() is still None")
        elif rec.tier != "interpreter" or rec.fallback_from != "auto":
            r.fail("the record names the interpreter + auto fallback", repr(rec))
        elif "_CgContinue" not in (rec.reason or ""):
            r.fail("the record's reason names the _CgContinue class", repr(rec))
        else:
            r.ok(f"recorded and non-silent: {rec!r}")
    finally:
        compiled_mod.run_auto = orig


def test_trk141_ordinary_decline_unchanged(r: SubTestResult):
    """An ORDINARY exception (not `_CgBreak`/`_CgContinue`) must NOT gain a
    tier_trace record from this fix — the settling text is explicit that this is
    never treated as an ordinary decline, i.e. the new behaviour must stay
    scoped to the two control-flow classes."""
    print("\n--- TRK-141: an ordinary exception still leaves no tier_trace record ---")
    bindings = _cg_break_bindings()
    orig = tex_engine.execute_compiled

    def _raise_ordinary(*a, **kw):
        raise RuntimeError("unrelated compile failure")

    tex_engine.execute_compiled = _raise_ordinary
    try:
        tier_trace.reset()
        result = tex_engine.cook(_CODE, dict(bindings), device_mode="cpu",
                                 compile_mode="torch_compile")
        rec = tier_trace.last()
        if result is None or "OUT" not in result.outputs:
            r.fail("torch_compile still serves a result on an ordinary decline",
                  "cook() returned no OUT")
        elif rec is not None:
            r.fail("an ordinary decline stays unrecorded (unchanged behaviour)",
                  repr(rec))
        else:
            r.ok("ordinary decline behaviour unchanged: tier_trace.last() is still None")
    finally:
        tex_engine.execute_compiled = orig
