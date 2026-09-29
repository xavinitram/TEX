"""Compiled-tier promotion (TRIAL) support — SPLIT (v0.50.0 Phase C, K0, R2#3).

Split mechanically out of `compiled.py` (the SPLIT-47/STR-7 pattern: every body below is
byte-identical to the code it replaced there — AGENTS.md §"Trades to REFUSE", mechanical
moves only, never an "improvement" mid-move). `compiled.py` reached the 2000-line hard
budget with no floor and no split plan (R2#3); this is its first domain cut, following the
exact shape its own history section already documents for `compiled_capability.py` /
`compiled_exec_support.py`.

This module owns AUTOSAFE-50's (TRK-223) promotion-step machinery: the in-flight TRIAL
invocation bookkeeping (`_trial_futures`), the cumulative visibility counters
(`_promotion_stats`, `promotion_stats()`, `_reset_promotion_stats_for_test()`), the bounded-
wait budget (`_TRIAL_WAIT_BUDGET_S`/`_TRIAL_POLL_SLICE_S`), and the submit/await pair
(`_submit_trial`/`_await_trial`) `run_auto`'s own TRIAL branch (still in `compiled.py`)
calls. `compiled.py` imports this module at its own top level and re-exports every name
below, so a bare call from within `run_auto`/`_drain_bg_for_test` resolves through
`compiled.py`'s own module globals exactly as before the move (the ROUTE-45 shape SPLIT-E
used) — `_trial_futures`/`_promotion_stats` are module-level MUTABLE stores (dict), so a
plain re-export keeps every in-place mutation (`_trial_futures[cache_key] = ...`,
`_promotion_stats["failed"] += 1`) visible from both this module and `compiled.py`'s own
call sites (the same rule SPLIT-47 already used for `_deferred_ev`/`_warnings_shown`).

This module reaches back into `compiled.py` for the names that stay there
(`_canon_device`, `_compiled_cache`, `_pool_for`, `_mark_pool_busy`, `_mark_pool_free`)
lazily, inside each function body that needs them — the same posture `compiled_capability.py` already uses for
`compiled._backend_status`/`compiled._setup_msvc_env` — so this module never imports
`compiled.py` at its own module scope and there is no load-time cycle."""
from __future__ import annotations

import concurrent.futures
import time as _time

import torch

from .compiled_exec_support import _contiguous_bindings, _is_transient_failure, _timed
from .host import _cancel_check  # SCHED-3 seam

# AUTOSAFE-50 (TRK-223): the TRIAL tier's first REAL invocation of a freshly-promoted
# compiled callable, in flight, keyed by cache_key -- the counterpart to `compiled._bg_futures`
# for the *promotion* step rather than the compile-wrap/warm step. See `_submit_trial`/
# `_await_trial` below.
_trial_futures: dict = {}

# AUTOSAFE-50: visible, best-effort counters for the promotion path (tier_trace already
# carries a per-cook reason string; this is the cumulative-count counterpart a host or a
# test can read without scraping `tier_trace.recent()`). Never gates behaviour.
_promotion_stats = {"bounded": 0, "failed": 0}


# Cache keys whose LAST failure looked transient (out of memory, a dead or abandoned worker) as
# opposed to a fact about the program. A verdict recorded from such a failure applies to this
# process only: persisting it would pin the program to the safe tier for every later session.
_transient_failed: set = set()


def _note_failure(cache_key, exc) -> None:
    """Remember that the failure just seen under `cache_key` is not a property of the program."""
    if _is_transient_failure(exc):
        _transient_failed.add(cache_key)


def _durable_failure(cache_key) -> bool:
    """True when a failure under `cache_key` may be persisted as a REJECTED verdict; False
    (consuming the note) when it was transient."""
    if cache_key in _transient_failed:
        _transient_failed.discard(cache_key)
        return False
    return True


def promotion_stats() -> dict:
    """AUTOSAFE-50: a copy of the cumulative promotion counters -- how many cooks
    deferred an in-flight TRIAL invocation past the bounded wait ("bounded"), and how
    many times a background compile or a TRIAL invocation ended in failure ("failed").
    Read-only for hosts/tests; tests reset with `_reset_promotion_stats_for_test`."""
    return dict(_promotion_stats)


def _reset_promotion_stats_for_test() -> None:
    _promotion_stats["bounded"] = 0
    _promotion_stats["failed"] = 0


# AUTOSAFE-50: the cook thread's own budget for waiting on an in-flight TRIAL job before
# giving up FOR THIS COOK and returning the safe (codegen) tier instead -- never the
# promotion's own deadline (the background job keeps running regardless; a later cook's
# poll is near-free once it is done). Small on purpose: the common case (an artifact the
# background warm already exercised) resolves inside one or two slices, and a genuinely
# slow real invocation (TRK-223) then costs this bound, not its own full duration, on the
# cook thread. Chosen in the same family as AUTO-47's probe bound (500 ms) and
# PREWARM-481's measured heartbeat bound (~120 ms) -- an order of magnitude tighter than
# either, because this wait sits on the INTERACTIVE cook path itself, not a one-time
# background warm.
_TRIAL_WAIT_BUDGET_S = 0.02
_TRIAL_POLL_SLICE_S = 0.005


def _submit_trial(cache_key, program, bindings, type_map, device,
                  latent_channel_count, output_names, device_type,
                  scale: float | None = None) -> bool:
    """AUTOSAFE-50 (TRK-223): submit the TRIAL tier's first REAL invocation of a
    freshly-promoted compiled callable as a background job instead of running it
    synchronously on the cook thread. A program that reaches TRIAL and gets a cache hit
    can still pay a slow, GIL-holding first invocation of the freshly-compiled callable
    (this ask's own repro: a deterministic stand-in; a real box may pay this from a
    fresh CUDA-graph capture at the real cook's own tensor addresses, a guard mismatch
    against the warm clone, or anything else `_try_compile`'s lazy wrap did not force) --
    exactly the class `_submit_bg_compile`'s `warm_call` already backgrounds for the
    WRAP step; this backgrounds the TRIAL step the same way, on the SAME dedicated
    `_WARM_POOL` (C1's isolation: never share a worker with a plain wrap-only submit on
    `_COMPILE_POOL`, and never spin up a new pool for this).

    Returns True once a job is in flight (submitting now, or already was) -- the caller
    (`run_auto`) polls it with `_await_trial`, bounded, never blocking the cook thread past
    a small budget. Idempotent per cache_key: a second call while one is already running
    is a no-op that returns True without resubmitting."""
    if cache_key in _trial_futures:
        return True
    from .compiled import (_canon_device, _compiled_cache, _pool_for,
                           _mark_pool_busy, _mark_pool_free)
    contiguous = _contiguous_bindings(bindings, _canon_device(device))

    def _worker():
        # K5 (v0.50.0 Phase C, B3#4): mark/clear this shared pool's busy window around
        # the real invocation, the same way `_submit_bg_compile`'s job does -- a TRIAL
        # invocation that never returns must not silently poison every OTHER
        # fingerprint's future submissions to `_WARM_POOL` forever.
        busy_token = _mark_pool_busy("warm")
        try:
            with torch.inference_mode():
                from .lru_util import lru_get
                entry = lru_get(_compiled_cache, cache_key)
                if entry is None:
                    return None, None
                compiled_fn, _b = entry
                call = lambda: compiled_fn(program, contiguous, type_map, device,
                                           latent_channel_count, output_names, scale=scale)
                return _timed(call, device_type)
        finally:
            _mark_pool_free("warm", busy_token)

    try:
        _trial_futures[cache_key] = _pool_for("warm").submit(_worker)
        return True
    except Exception:
        return False


def _await_trial(cache_key, cancel=None):
    """AUTOSAFE-50: bounded, cancellable poll of the in-flight TRIAL job `_submit_trial`
    started for `cache_key`. Returns one of:
      ("ready", (res, ms))  the invocation finished; `res` is the compiled tier's output.
      ("failed", None)      it finished by raising, or produced no artifact to run.
      ("pending", None)     still running after `_TRIAL_WAIT_BUDGET_S` -- unchanged, still
                            in `_trial_futures`; a LATER cook polls again (near-zero cost
                            once it is actually done: `Future.result(timeout=~0)`).
      ("absent", None)     no job was ever submitted for this key (caller bug/race).

    Waits in short slices (`_TRIAL_POLL_SLICE_S`) rather than one call so a supplied
    `cancel` token is checked promptly (SCHED-3: a cancel aborts and is never swallowed --
    `_cancel_check` raises `CookCancelled`, left to propagate) instead of only after the
    whole budget elapses, and so a job that is already done (the common case: the
    background warm already exercised this exact call) is picked up on the very first,
    near-instant slice -- the same cook it was submitted on, same as the synchronous call
    this replaces used to do for a fast artifact."""
    fut = _trial_futures.get(cache_key)
    if fut is None:
        return "absent", None
    deadline = _time.monotonic() + _TRIAL_WAIT_BUDGET_S
    while True:
        _cancel_check(cancel)   # SCHED-3: propagate CookCancelled, never swallowed
        remaining = deadline - _time.monotonic()
        if remaining <= 0:
            return "pending", None
        try:
            result = fut.result(timeout=min(_TRIAL_POLL_SLICE_S, remaining))
        except concurrent.futures.TimeoutError:
            continue
        except Exception as exc:
            _trial_futures.pop(cache_key, None)
            _note_failure(cache_key, exc)
            # Mirrors _run_cached_compiled's own crash handling: demote, reset dynamo on
            # THIS (the calling) thread -- dynamo state is process-global (DO-NOT-TOUCH).
            from .compiled import _compiled_cache
            _compiled_cache.pop(cache_key, None)
            try:
                torch._dynamo.reset()
            except Exception:
                pass
            return "failed", None
        _trial_futures.pop(cache_key, None)
        if result is None or result[0] is None:
            return "failed", None
        return "ready", result
