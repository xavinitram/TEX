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
import threading
from collections import OrderedDict

_MEMO_MAX = 512

# K4 (v0.50.0 Phase C, B3#3): `begin_attempt`'s check-then-add across `_memo`/`_pending`
# is not atomic on its own -- two separate Python statements, no lock. The SAME
# fingerprint/device/precision key can reach `_try_compile` from either
# `execute_compiled`'s `_compile_and_run` (`compiled._COMPILE_POOL`) or
# `_submit_bg_compile`'s `_compile_and_maybe_warm` (`compiled._WARM_POOL`) -- two
# independent, genuinely concurrent single-worker pools -- if a workflow mixes
# `compile_mode` across cooks of the same program, or two nodes share a fingerprint.
# CPython can switch threads between the `in` check and the `.add()`, so both pools'
# workers can observe "not yet pending", both add it, and both proceed to call
# `torch.compile()`/run the real invocation for the SAME key concurrently -- the exact
# hazard `execute_compiled`'s own comment names (dynamo's C++ TLS is corrupted by a
# concurrent trace on two threads). Held only across the tiny check-and-set below, never
# across `_try_compile` itself (that call already runs off this module, on whichever
# pool's worker granted the attempt).
_lock = threading.Lock()

#: composite key (see `_key`, below) -> True (a real torch.compile backend was produced) |
#: False (the whole backend cascade failed) -- terminal once set; never flipped (the
#: "paid once" contract).
_memo: "OrderedDict[str, bool]" = OrderedDict()

#: composite keys mid-attempt: `begin_attempt` granted the ONE fall-through and is waiting
#: for the caller's `resolve_attempt`. Session-only (never persisted) -- a process restart
#: with a key still pending here just tries again, exactly like `_bg_futures`/
#: `_trial_futures` never surviving a restart either.
_pending: set[str] = set()


def _key(fp: str | None, device_type: str, precision: str) -> str | None:
    """K3 (v0.50.0 Phase C, R4 F3): the composite memo key -- fingerprint ALONE is not
    enough. What this module memoizes (whether `_try_compile`'s real backend cascade
    succeeds) is exactly as device/precision-dependent as `compiled.py`'s own
    `cache_key = (fingerprint, device_type, precision)` and `autotier.make_key` already
    model -- this module's own header names a real CPU-inductor-needs-MSVC-always-fails
    failure mode. `graphed._capturable_memo`'s fingerprint-only key (the shape this
    module borrowed) is correct THERE because CUDA-graph capturability is a static,
    box-independent AST property; "does a real compile succeed" is not.

    A single `|`-joined string, not a tuple: `_memo`/`_pending` stay string-keyed (the
    exact shape `warm_state.py` already reads/writes `_memo` by by name -- e.g.
    `note_fncalls_update`'s `_memo.get(fp)` -- so widening the semantic key needs no
    change there, and the on-disk `fncalls_compile` JSON object in `warm_state.json`
    stays a flat `{key: bool}` map, not a list-of-records restructuring). Safe to join
    on `|`: `device_type` ("cpu"/"cuda"[:N]) and `precision` ("fp32"/"fp16") never
    contain it, and a fingerprint is a hex digest."""
    if fp is None:
        return None
    return f"{fp}|{device_type}|{precision}"


def verdict(fp: str | None, device_type: str, precision: str) -> bool | None:
    """`None` (never resolved this fingerprint/device/precision), `True` (compile it), or
    `False` (use the codegen-only eager adapter -- a prior attempt's every backend
    failed). Adopts a persisted verdict on first miss (CACHE-3's own pattern,
    `graphed.capture`'s `_capturable_memo.get` mirrored here) so a key resolved in an
    EARLIER process is not re-attempted in this one."""
    key = _key(fp, device_type, precision)
    if key is None:
        return None
    v = _memo.get(key)
    if v is None:
        try:
            from . import warm_state as _ws
            _ws.ensure_loaded()
            v = _memo.get(key)
        except Exception:
            pass
    else:
        _memo.move_to_end(key)
    return v


def begin_attempt(fp: str | None, device_type: str, precision: str) -> bool:
    """Grant the ONE remembered fall-through attempt for this (fingerprint, device,
    precision). Returns True exactly once per key (until `resolve_attempt` settles it, or
    forever if it never does -- `_try_compile`'s two callers always call `resolve_attempt`
    on every path out, including every exception path, so this should not linger); False
    for a key already resolved, already pending, or a `None` fingerprint -- a program with
    no fingerprint (e.g. an uncached probe call) has no key to remember a verdict against,
    so it keeps today's exact behaviour (the always-safe codegen-only adapter, never
    attempted) rather than retrying forever with nothing to show for it.

    K4: the check-then-add is now atomic across every caller (held only across these two
    lines, never across the caller's own `_try_compile` attempt) -- two pools racing the
    SAME key can no longer both observe "not yet pending" and both proceed."""
    key = _key(fp, device_type, precision)
    if key is None:
        return False
    with _lock:
        if key in _memo or key in _pending:
            return False
        _pending.add(key)
        return True


def resolve_attempt(fp: str | None, device_type: str, precision: str,
                    backend: str | None) -> None:
    """The caller of `_try_compile` reports what it got back for a (fingerprint, device,
    precision) `begin_attempt` granted the fall-through to: `backend` is `_try_compile`'s
    own second return value (a backend name string for a real torch.compile artifact,
    `None` for the codegen-only eager adapter or an overall "no backend available"). A
    no-op for any key NOT currently pending -- every ordinary call (no `_has_fn_calls`, or
    a key that was never granted a fall-through) passes through here for free."""
    key = _key(fp, device_type, precision)
    if key is None or key not in _pending:
        return
    _pending.discard(key)
    record(key, backend is not None)


def record(key: str, ok: bool) -> None:
    """Settle `key`'s verdict once, terminally, and persist it. Idempotent: a key already
    memoized keeps its FIRST verdict -- this should not be called twice for the same `key`
    in ordinary operation (`resolve_attempt`'s `_pending` guard already prevents it), but a
    test or a race must never flip an already-settled verdict. `key` is the already-composed
    string from `_key()` (or an equivalent test-composed one) -- this function does not
    itself take separate fp/device/precision components."""
    if key in _memo:
        return
    _memo[key] = bool(ok)
    while len(_memo) > _MEMO_MAX:
        _memo.popitem(last=False)
    try:
        from . import warm_state as _ws
        _ws.note_fncalls_update(key)
    except Exception:
        pass


def adopt_persisted(key: str | None, ok) -> None:
    """`warm_state.load()` calls this while merging a snapshot/journal entry. `setdefault`
    semantics: a verdict this session already reached (fresher) always wins over a stale
    disk verdict -- the same rule `warm_state.load`'s own `adopt()` uses for
    `graphed._capturable_memo`. `key` is the already-composed string from the persisted
    JSON, not a bare fingerprint."""
    if key is None or ok is None:
        return
    _memo.setdefault(key, bool(ok))


def snapshot_items():
    """What `warm_state._snapshot()` persists: every terminal verdict, composite key ->
    bool. The composite key is already a flat string (`_key()`), so the persisted JSON
    object stays exactly the same shape it was before K3 (fp -> bool) -- only what the
    string spells changed."""
    return dict(_memo)


def reset_for_test() -> None:
    """Test hook: forget every verdict and every in-flight attempt."""
    _memo.clear()
    _pending.clear()
