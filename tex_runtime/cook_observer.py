"""tex_runtime/cook_observer.py — OBSERVER-46: the supported cook-observer seam.

ROUTE-45's audit found the failure mode this module exists to retire: `cook_stage_list`
and `boundary_lineage_key` moved to `tex_chain` (NEG-2) and are re-exported from
`tex_engine`, but `tex_chain.cook_fused_cached` calls its OWN module's `cook_stage_list`
and `boundary_lineage_key` — a host that wraps `tex_engine.cook_stage_list` to count cooks
never sees those internal calls. A re-export plus an internal self-call defeats any wrap on
the old name, and the next such move would defeat it again. This module is the alternative:
a host registers a callback here once, instead of monkey-patching a name that might move.

**The six cook entry points**, each calling `enter`/`leave` (below) at its own top and
bottom: `tex_engine.run`, `tex_engine.cook`, `tex_chain.cook_stage_list`,
`tex_chain.cook_fused_cached`, `tex_checkpoint.cook_checkpointed` and
`tex_chain.boundary_lineage_key`. A callback registered with `register` is called as
`cb(entry, thread)` once — see "Once", below — for every one of them, by name (`entry` is
the plain function name: `"run"`, `"cook"`, `"cook_stage_list"`, `"cook_fused_cached"`,
`"cook_checkpointed"`, `"boundary_lineage_key"`).

**"Once" means once per EXTERNAL call, not once per function body.** `cook` calls
`run(prepare(...))`; `cook_fused_cached` calls `cook_stage_list` (up to three times, on the
CACHE-6 hot path) and `boundary_lineage_key`; `cook_checkpointed` calls both of those too
(via `tex_engine.cook_stage_list`/`tex_engine.boundary_lineage_key`, module-lookup, per
cut). None of that nesting is a second cook — it is one host-observed cook recursing
through some of the six functions underneath. `enter`/`leave` track a per-thread depth: the
OUTERMOST call among the six on a thread notifies, every call nested inside it (by any of
the six, in any combination) shares that single notification. This is the shape a
count-once audit wants — an embedding host's audit that enforces "one cook-queue
worker thread" by counting cooks, which is exactly the property a double-notified nested
call would break. A host that wants the nesting depth instead of the collapsed count is not
served by this seam as specified; nothing here prevents building that separately.

**Zero cost when nothing is registered.** Every call site guards `enter`/`leave` behind a
plain truthiness check on `_callbacks` (a dict) — `if _callbacks: enter(name)` — so an
unregistered process pays one dict-truthiness check per entry point and never calls into
this module at all. This mirrors the `_profile.enabled()` gate `tex_engine.run` already
takes for the same reason (see that module).

**Callback exceptions never break a cook.** `enter` catches anything a callback raises,
and warns once per PROCESS (never once per callback, never once per cook — the same
warn-once posture `tex_runtime.noise` already uses for its own kernel-block warning) so a
misbehaving observer cannot flood a log or, worse, abort a cook that would otherwise have
completed.

**Thread-safe registration.** `register`/`unregister` and the dispatch loop all take the
same lock; the lock is held only to snapshot or mutate the callback dict, never across a
callback invocation, so one slow or blocked observer cannot stall another thread's
`register`/`unregister` call (it CAN, by design, still be slow for the cook thread that is
calling it — a callback is expected to be quick, the same expectation any observer/listener
API makes of its callback).
"""
from __future__ import annotations

import threading
import warnings
from typing import Callable

#: `cb(entry, thread)` — `entry` is one of the six names in the module docstring; `thread`
#: is the `threading.Thread` that is doing the cooking (`threading.current_thread()`).
CookObserverCallback = Callable[[str, "threading.Thread"], None]

#: The six cook entry points this seam covers, spelled once so a docstring or a test can
#: name "the six" without retyping the list by hand.
ENTRY_POINTS = (
    "run", "cook", "cook_stage_list", "cook_fused_cached",
    "cook_checkpointed", "boundary_lineage_key",
)

_lock = threading.Lock()
_callbacks: dict[int, CookObserverCallback] = {}
_next_handle = 0

_local = threading.local()   # per-thread reentrancy depth; see `enter`/`leave`
_warned = False               # process-wide "a callback raised" warn-once latch (no lock:
#                                a benign double-warn under a race is not worth one)


def register(cb: CookObserverCallback) -> int:
    """Register `cb` to be notified once at each external cook (see the module docstring
    for exactly what "once" means). Returns an opaque handle for `unregister`. A callback
    may be registered more than once (each registration gets its own handle and is called
    once per notification, same as any other observer list)."""
    global _next_handle
    with _lock:
        handle = _next_handle
        _next_handle += 1
        _callbacks[handle] = cb
    return handle


def unregister(handle: int) -> None:
    """Remove a callback registered by `register`. A stale, foreign or already-removed
    handle is a silent no-op — a host's teardown never has to guard this call."""
    with _lock:
        _callbacks.pop(handle, None)


def _dispatch(entry: str) -> None:
    """Call every currently-registered callback with (`entry`, the calling thread). Only
    ever reached from `enter` once `_callbacks` is already known non-empty, and only for the
    outermost of a nest — see `enter`."""
    global _warned
    with _lock:
        cbs = list(_callbacks.values())
    thread = threading.current_thread()
    for cb in cbs:
        try:
            cb(entry, thread)
        except Exception as exc:                     # a callback's bug is never the cook's
            if not _warned:
                _warned = True
                # O1 (v0.46, FIX-OBSROUTE): `warnings.warn` itself RAISES under
                # warnings-as-errors (`-W error`, `simplefilter("error")`, strict pytest) —
                # a misbehaving callback would then take down the cook that was calling
                # `enter()`, which is exactly the outcome this whole catch exists to prevent.
                # The report is best-effort like the callback dispatch it is reporting on:
                # swallow whatever raising it produces, one report per process.
                try:
                    warnings.warn(
                        f"TEX: a cook_observer callback raised "
                        f"({type(exc).__name__}: {exc}); ignoring it for the rest of this "
                        "process (the cook that triggered it is unaffected). Reported once "
                        "per process, not once per callback or per cook.",
                        RuntimeWarning, stacklevel=3)
                except Exception:
                    pass


def enter(entry: str) -> None:
    """Call at the top of one of the six cook entry points, guarded by the caller with
    `if _callbacks:` (see the module docstring's zero-cost note — this function itself does
    no such check, so calling it unconditionally would cost one call+attribute-lookup even
    when nothing is registered). Notifies every registered callback with `entry` ONLY when
    this is the outermost call among the six on the current thread; a nested call
    increments the depth counter and returns without dispatching. Always pair with `leave()`
    in a `finally`, so an exception out of the cook body still balances the depth.

    O2 (v0.46, FIX-OBSROUTE): depth is incremented BEFORE dispatch, not after. A callback
    that itself triggers a cook on the SAME thread (a reentrant observer) must see depth
    already at 1 when its own nested `enter()` runs — otherwise the nested call reads the
    still-zero depth `_dispatch` was called under and notifies a second time for what the
    host's count-once contract considers one cook."""
    depth = getattr(_local, "depth", 0) + 1
    _local.depth = depth
    if depth == 1:
        _dispatch(entry)


def leave() -> None:
    """Pair every guarded `enter()` call, unconditionally, in a `finally`."""
    _local.depth -= 1


class _Scope:
    """The object `scope()` returns — see `scope()` for the contract. Not constructed
    directly."""
    __slots__ = ("_entry", "_active")

    def __init__(self, entry: str) -> None:
        self._entry = entry
        self._active = False

    def __enter__(self) -> "_Scope":
        # O3: ONE snapshot of "is anything registered?" per outermost call, taken here
        # rather than re-read by `__exit__` — the same race B4#3 named for the 6
        # copy-pasted blocks (a register/unregister landing between a block's own enter
        # guard and its leave guard) is closed the same way each of them closed it: read
        # `_callbacks` once, act on that one answer for the whole scope.
        self._active = bool(_callbacks)
        if self._active:
            enter(self._entry)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._active:
            leave()
        return False


def scope(entry: str) -> "_Scope":
    """A context manager replacing the 6 copy-pasted
    `_obs_active = bool(_callbacks); if _obs_active: enter(name)` / `finally: if _obs_active:
    leave()` blocks at `tex_engine.run`/`cook`, `tex_chain.cook_stage_list`/
    `cook_fused_cached`/`boundary_lineage_key` and `tex_checkpoint.cook_checkpointed`:

        with cook_observer.scope("run"):
            ...cook body...

    Same zero-added-notification-when-unregistered contract as the hand-rolled blocks
    (`__enter__` reads `_callbacks` once and skips `enter()` entirely when it is empty), and
    the same `enter()`-never-raises guarantee (O1) means `__enter__` cannot leave `_active`
    set without having actually entered, so `__exit__`'s `leave()` always pairs correctly
    even if a callback misbehaves."""
    return _Scope(entry)
