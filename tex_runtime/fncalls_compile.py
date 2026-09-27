"""tex_runtime/fncalls_compile.py — COMPILETRY-50 (D1).

Per-fingerprint memo replacing `compiled._try_compile`'s former BLANKET `_has_fn_calls`
gate: a program whose codegen'd function calls a non-inlined stdlib builtin
(`erode`/`dilate`/`bilateral_filter`/`gauss_blur`, today) used to be handed the codegen-only
eager adapter unconditionally, on the AST alone, never asking Dynamo the question. Three of
those four builtins already trace clean by measurement; the blanket gate never found out.

The new rule ("D1 = try-and-fall-back, remembered"): a fingerprint gets exactly ONE real
`torch.compile()` attempt. This
module remembers the outcome -- `True` (a real backend was produced) or `False` (every
backend in `_try_compile`'s own cascade failed) -- so it is paid at most once per fingerprint,
ever, including across a process restart (persisted via `warm_state.py`'s existing
snapshot+journal, NOT a new store -- the same "generalizes the autotier.json pattern" this
module's sibling `graphed._capturable_memo` already uses).

The attempt itself needs NO new background machinery: `_try_compile` is already only ever
invoked from inside a worker already submitted to `compiled._COMPILE_POOL`/`_WARM_POOL`
(AUTOSAFE-50's own off-thread promotion pools) by its two existing callers
(`execute_compiled`'s `_compile_and_run`, `_submit_bg_compile`'s `_compile_and_maybe_warm`) --
never on the cook thread. This module only decides WHETHER `_try_compile` should fall through
to the real attempt (`begin_attempt`) and records what its caller observed
(`resolve_attempt`); it submits nothing itself.
"""
from __future__ import annotations
from collections import OrderedDict

_MEMO_MAX = 512

#: fingerprint -> True (a real torch.compile backend was produced) | False (the whole
#: backend cascade failed) -- terminal once set; never flipped (the "paid once" contract).
_memo: "OrderedDict[str, bool]" = OrderedDict()

#: fingerprints mid-attempt: `begin_attempt` granted the ONE fall-through and is waiting for
#: the caller's `resolve_attempt`. Session-only (never persisted) -- a process restart with a
#: fingerprint still pending here just tries again, exactly like `_bg_futures`/`_trial_futures`
#: never surviving a restart either.
_pending: set[str] = set()


def verdict(fp: str | None) -> bool | None:
    """`None` (never resolved this fingerprint), `True` (compile it), or `False` (use the
    codegen-only eager adapter -- a prior attempt's every backend failed). Adopts a
    persisted verdict on first miss (CACHE-3's own pattern, `graphed.capture`'s
    `_capturable_memo.get` mirrored here) so a fingerprint resolved in an EARLIER process
    is not re-attempted in this one."""
    if fp is None:
        return None
    v = _memo.get(fp)
    if v is None:
        try:
            from . import warm_state as _ws
            _ws.ensure_loaded()
            v = _memo.get(fp)
        except Exception:
            pass
    else:
        _memo.move_to_end(fp)
    return v


def begin_attempt(fp: str | None) -> bool:
    """Grant the ONE remembered fall-through attempt for `fp`. Returns True exactly once
    per fingerprint (until `resolve_attempt` settles it, or forever if it never does --
    `_try_compile`'s two callers always call `resolve_attempt` on every path out, including
    every exception path, so this should not linger); False for a fingerprint already
    resolved, already pending, or `None` -- a program with no fingerprint (e.g. an uncached
    probe call) has no key to remember a verdict against, so it keeps today's exact
    behaviour (the always-safe codegen-only adapter, never attempted) rather than retrying
    forever with nothing to show for it."""
    if fp is None or fp in _memo or fp in _pending:
        return False
    _pending.add(fp)
    return True


def resolve_attempt(fp: str | None, backend: str | None) -> None:
    """The caller of `_try_compile` reports what it got back for a fingerprint
    `begin_attempt` granted the fall-through to: `backend` is `_try_compile`'s own second
    return value (a backend name string for a real torch.compile artifact, `None` for the
    codegen-only eager adapter or an overall "no backend available"). A no-op for any `fp`
    NOT currently pending -- every ordinary call (no `_has_fn_calls`, or a fingerprint
    that was never granted a fall-through) passes through here for free."""
    if fp is None or fp not in _pending:
        return
    _pending.discard(fp)
    record(fp, backend is not None)


def record(fp: str, ok: bool) -> None:
    """Settle `fp`'s verdict once, terminally, and persist it. Idempotent: a fingerprint
    already memoized keeps its FIRST verdict -- this should not be called twice for the
    same `fp` in ordinary operation (`resolve_attempt`'s `_pending` guard already prevents
    it), but a test or a race must never flip an already-settled verdict."""
    if fp in _memo:
        return
    _memo[fp] = bool(ok)
    while len(_memo) > _MEMO_MAX:
        _memo.popitem(last=False)
    try:
        from . import warm_state as _ws
        _ws.note_fncalls_update(fp)
    except Exception:
        pass


def adopt_persisted(fp: str | None, ok) -> None:
    """`warm_state.load()` calls this while merging a snapshot/journal entry. `setdefault`
    semantics: a verdict this session already reached (fresher) always wins over a stale
    disk verdict -- the same rule `warm_state.load`'s own `adopt()` uses for
    `graphed._capturable_memo`."""
    if fp is None or ok is None:
        return
    _memo.setdefault(fp, bool(ok))


def snapshot_items():
    """What `warm_state._snapshot()` persists: every terminal verdict, fp -> bool."""
    return dict(_memo)


def reset_for_test() -> None:
    """Test hook: forget every verdict and every in-flight attempt."""
    _memo.clear()
    _pending.clear()
