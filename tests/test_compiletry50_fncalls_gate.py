"""
COMPILETRY-50 (v0.50, D1) — `compiled._try_compile`'s former BLANKET `_has_fn_calls` gate.

BEFORE this ask: a program whose codegen'd function calls a non-inlined stdlib builtin
(`erode`/`dilate`/`bilateral_filter`/`gauss_blur`) was handed the codegen-only eager adapter
UNCONDITIONALLY — the AST alone (`_has_fn_calls`) decided, `torch.compile` was never asked.
Measurement found that 3 of the 4 affected builtins already trace clean under Dynamo; the
blanket gate never found out.

AFTER: `tex_runtime.fncalls_compile` grants each fingerprint exactly ONE real fall-through
attempt, remembers the outcome (True = a working backend was produced, False = the whole
backend cascade failed), and persists it via `warm_state.py` (NOT a new store) so it is paid
at most once per fingerprint, ever — including across a process restart. A failure falls back
to exactly today's codegen-only path and is visible (`tier_trace.record` +
`compiled._promotion_stats["failed"]`).

PORTABILITY. CPU only, no real `torch.compile`/Inductor/Triton anywhere in this file — this
box's CPU Inductor needs a C++ compiler this box does not have (`cl` not found, reproduced
manually), so every test here stands in for `torch.compile` and `_select_backend` with
deterministic fakes, mirroring `test_compile_a_toolchain.py`'s own CC-5 rationale ("the
mechanism under test... does not need a real backend to prove"). `_try_compile` is exercised
directly with a monkeypatched `_get_or_make_codegen_fn`, so no real TEX program/codegen
pipeline is needed either — only the ONE attribute `_try_compile` actually reads,
`cg_fn._has_fn_calls`.
"""
import pytest

from helpers import *  # noqa: F401,F403
from TEX_Wrangle.tex_runtime import compiled as C
from TEX_Wrangle.tex_runtime import fncalls_compile as FC
from TEX_Wrangle.tex_runtime import warm_state as WS
from TEX_Wrangle.tex_runtime import tier_trace


def _fake_cg_fn(has_fn_calls=True):
    def _fn(*a, **k):
        return None
    _fn._has_fn_calls = has_fn_calls
    return _fn


def _stand_in_select_backend(name="stand_in", tries=1):
    """A `_select_backend` stand-in that offers `name` exactly `tries` time(s), then
    exhausts (returns None) -- mirrors the real cascade's own termination shape
    (`compiled_capability._select_backend`: each failure marks that backend False, so the
    NEXT call finds no candidate) without depending on `_backend_status`'s real state."""
    state = {"left": tries}

    def _fake(device_type):
        if state["left"] <= 0:
            return None
        state["left"] -= 1
        return name
    return _fake


@pytest.fixture(autouse=True)
def _isolated_fncalls_state(monkeypatch):
    """Every test in this file gets a clean `fncalls_compile` table and a scratch
    `TEX_CACHE_DIR` (so warm_state persistence round-trips don't touch a real cache dir or
    leak into another test), restored on the way out -- the same shape
    `tex_testkit.cold_engine_state` gives other v050 files, sized to just this module's
    two extra globals (`fncalls_compile`, `warm_state`)."""
    with cold_engine_state():
        FC.reset_for_test()
        WS._reset_for_test()
        yield
        FC.reset_for_test()
        WS._reset_for_test()


# ── fncalls_compile: the pure memo, no torch.compile involved ──────────────────────────

def test_verdict_none_until_recorded():
    assert FC.verdict("fp-a", "cpu", "fp32") is None
    FC.record(FC._key("fp-a", "cpu", "fp32"), True)
    assert FC.verdict("fp-a", "cpu", "fp32") is True


def test_record_is_terminal_first_writer_wins():
    key = FC._key("fp-b", "cpu", "fp32")
    FC.record(key, False)
    FC.record(key, True)   # a later, contradicting write must not flip it
    assert FC.verdict("fp-b", "cpu", "fp32") is False


def test_begin_attempt_granted_once_then_pending():
    assert FC.begin_attempt("fp-c", "cpu", "fp32") is True
    assert FC.begin_attempt("fp-c", "cpu", "fp32") is False   # already pending -- no second grant
    FC.resolve_attempt("fp-c", "cpu", "fp32", "inductor")
    assert FC.verdict("fp-c", "cpu", "fp32") is True
    assert FC.begin_attempt("fp-c", "cpu", "fp32") is False   # already resolved -- no grant either


def test_begin_attempt_none_fingerprint_never_granted():
    """A program with no fingerprint (e.g. an uncached probe) has no key to remember a
    verdict against -- it must keep taking the always-safe path, never the fall-through,
    exactly like every fingerprint-less call did before this ask."""
    assert FC.begin_attempt(None, "cpu", "fp32") is False
    assert FC.verdict(None, "cpu", "fp32") is None


def test_resolve_attempt_noop_for_a_fingerprint_never_granted():
    """A fingerprint nobody called `begin_attempt` for (verdict already True/False, or
    ordinary non-fn-calls program) must not be recorded by a stray `resolve_attempt` --
    every real caller calls it unconditionally after every `_try_compile`, so this is the
    guard that keeps that safe for the overwhelming majority of calls."""
    FC.resolve_attempt("fp-never-pending", "cpu", "fp32", "inductor")
    assert FC.verdict("fp-never-pending", "cpu", "fp32") is None


def test_persists_across_a_simulated_restart():
    """Record a verdict, forget it (simulating a fresh process's empty in-memory table),
    then load from disk -- the CACHE-3 pattern this ask reuses rather than a new store."""
    key = FC._key("fp-restart", "cpu", "fp32")
    FC.record(key, True)
    WS.persist(force=True)
    FC.reset_for_test()
    WS._reset_for_test()
    assert key not in FC._memo   # nothing in memory yet (verdict() itself would
                                 # auto-load on a miss -- checked directly here)
    WS.load()
    assert FC.verdict("fp-restart", "cpu", "fp32") is True   # adopted from the warm_state snapshot


def test_journal_recovers_a_verdict_never_snapshotted():
    """ENG-13's crash-tight half: `note_fncalls_update` journals immediately (no throttle),
    so a verdict learned less than `_PERSIST_THROTTLE_SEC` before a simulated crash (never
    reaching `persist(force=True)`) is still recovered from the journal alone."""
    key = FC._key("fp-journal", "cpu", "fp32")
    FC.record(key, False)
    WS.note_fncalls_update(key)   # journals now; the throttled snapshot may not fire
    FC.reset_for_test()
    WS._reset_for_test()
    assert key not in FC._memo
    WS.load()
    assert FC.verdict("fp-journal", "cpu", "fp32") is False


# ── _try_compile: the gate itself ───────────────────────────────────────────────────────

def test_first_attempt_falls_through_and_succeeds(monkeypatch):
    """A fresh (never-seen) fingerprint whose codegen calls a stdlib builtin gets ONE real
    fall-through attempt -- RED at base (today's blanket gate never calls `torch.compile`
    at all for such a program); GREEN at head."""
    calls = {"n": 0}

    def _fake_torch_compile(fn, **kw):
        calls["n"] += 1
        return fn

    monkeypatch.setattr(C, "_select_backend", _stand_in_select_backend())
    monkeypatch.setattr(C.torch, "compile", _fake_torch_compile)
    monkeypatch.setattr(C, "_get_or_make_codegen_fn", lambda *a, **k: _fake_cg_fn())

    entry = C._try_compile("cpu", program=object(), type_map={}, fingerprint="fp-ok")
    C.fncalls_compile.resolve_attempt("fp-ok", "cpu", "fp32",
                                      entry[1] if entry is not None else None)

    assert entry is not None and entry[1] == "stand_in"
    assert calls["n"] == 1
    assert FC.verdict("fp-ok", "cpu", "fp32") is True


def test_adopted_true_verdict_skips_fall_through_bookkeeping(monkeypatch):
    """Once a fingerprint's verdict is True, a LATER call (e.g. after `_compiled_cache`
    was evicted/cleared, or a fresh process adopted the persisted verdict) skips the
    fall-through machinery entirely and goes straight to a real `torch.compile` attempt:
    exactly ONE `torch.compile` call, with no `begin_attempt`/`_pending` bookkeeping
    (no stale pending entry) and no codegen-only eager path."""
    calls = {"n": 0}

    def _fake_torch_compile(fn, **kw):
        calls["n"] += 1
        return fn

    monkeypatch.setattr(C, "_select_backend", _stand_in_select_backend())
    monkeypatch.setattr(C.torch, "compile", _fake_torch_compile)
    monkeypatch.setattr(C, "_get_or_make_codegen_fn", lambda *a, **k: _fake_cg_fn())

    key = FC._key("fp-warm", "cpu", "fp32")
    FC.record(key, True)   # simulates a verdict adopted from an earlier process
    entry = C._try_compile("cpu", program=object(), type_map={}, fingerprint="fp-warm")
    assert entry is not None and entry[1] == "stand_in"
    assert calls["n"] == 1
    assert key not in FC._pending   # never entered the fall-through bookkeeping


def test_failed_attempt_settles_false_and_falls_back(monkeypatch):
    """The ONE remembered attempt exhausting every backend in the cascade (CPU only ever
    offers one candidate, `compiled_capability._select_backend`) settles the fingerprint's
    verdict False -- `_try_compile` returns `None` overall here, exactly what it returned
    for ANY program before this ask when no backend is available (a pre-existing, already
    -visible-via-`_show_once`/`_backend_status` situation this ask does not change); the
    caller falls back to the plain interpreter, same as always. The NEW visibility this
    ask adds (`tier_trace` + `_promotion_stats["failed"]`) is for the ONGOING cost this
    settled verdict now avoids paying again -- proven by the next test, which is this same
    fingerprint's very next cook."""
    def _raising_compile(fn, **kw):
        raise RuntimeError("stand-in: no working backend")

    monkeypatch.setattr(C, "_select_backend", _stand_in_select_backend())
    monkeypatch.setattr(C.torch, "compile", _raising_compile)
    monkeypatch.setattr(C, "_get_or_make_codegen_fn", lambda *a, **k: _fake_cg_fn())

    entry = C._try_compile("cpu", program=object(), type_map={}, fingerprint="fp-fail")
    C.fncalls_compile.resolve_attempt("fp-fail", "cpu", "fp32",
                                      entry[1] if entry is not None else None)

    assert entry is None   # no backend at all -- the caller's own interpreter fallback
    assert FC.verdict("fp-fail", "cpu", "fp32") is False


def test_resolved_false_never_recompiles_and_stays_visible(monkeypatch):
    """Once a fingerprint is resolved False, a later call must NOT attempt `torch.compile`
    again (the "paid once" contract) but the fallback must still be reported every time it
    actually falls back — visibility is per-cook, not per-fingerprint."""
    def _never_call(fn, **kw):
        raise AssertionError("torch.compile must not be called for an already-failed fp")

    monkeypatch.setattr(C, "_select_backend", _stand_in_select_backend())
    monkeypatch.setattr(C.torch, "compile", _never_call)
    monkeypatch.setattr(C, "_get_or_make_codegen_fn", lambda *a, **k: _fake_cg_fn())

    key = FC._key("fp-known-bad", "cpu", "fp32")
    FC.record(key, False)
    before_failed = C.promotion_stats()["failed"]

    entry = C._try_compile("cpu", program=object(), type_map={}, fingerprint="fp-known-bad")

    assert entry is not None and entry[1] is None
    assert C.promotion_stats()["failed"] == before_failed + 1   # still reported this cook
    assert key not in FC._pending


def test_no_fn_calls_program_is_unaffected(monkeypatch):
    """An ordinary program (no stdlib calls) must never touch `fncalls_compile` at all --
    the gate this ask replaces only ever applied to `_has_fn_calls` programs."""
    calls = {"n": 0}

    def _fake_torch_compile(fn, **kw):
        calls["n"] += 1
        return fn

    monkeypatch.setattr(C, "_select_backend", _stand_in_select_backend())
    monkeypatch.setattr(C.torch, "compile", _fake_torch_compile)
    monkeypatch.setattr(C, "_get_or_make_codegen_fn",
                        lambda *a, **k: _fake_cg_fn(has_fn_calls=False))

    entry = C._try_compile("cpu", program=object(), type_map={}, fingerprint="fp-plain")

    assert entry is not None and entry[1] == "stand_in"
    assert calls["n"] == 1
    assert FC.verdict("fp-plain", "cpu", "fp32") is None   # never memoized -- this gate never ran
