"""
ENG-9 — a per-thread Interpreter pool, single-sourced.

The interpreter carries per-instance execution state (the scope stack, `_literal_cache`,
`_builtins_lru`) that is only reentrant under one-cook-at-a-time; a shared instance
corrupts under a branch-parallel executor (a two-thread cook mixes up programs). Each
pool hands every thread its own Interpreter and keeps a `weakref.WeakSet` of the live
ones so a memory sweep can clear them all — the thread-local holds the only strong ref,
so a dead thread's interpreter is collectable rather than pinned for the process lifetime.

Two pools exist (the engine's cook interpreter, `tex_engine`; and codegen's persistent
fallback, `compiled`); this class is the one implementation both instantiate, so the
lock discipline and the WeakSet-not-list rationale live in exactly one place.
"""
import threading
import weakref


class ThreadLocalInterpreterPool:
    """`get()` returns the calling thread's Interpreter (created on first use, lock-free
    on the hot path); `clear_all()` sweeps every live instance's tensor LRUs."""

    def __init__(self, factory):
        self._factory = factory              # () -> Interpreter
        self._tls = threading.local()
        self._all = weakref.WeakSet()        # live instances, for clear_all()
        self._lock = threading.Lock()        # insert-only: guards the registry add

    def get(self):
        inst = getattr(self._tls, "instance", None)
        if inst is None:
            inst = self._factory()
            self._tls.instance = inst
            with self._lock:
                self._all.add(inst)
        return inst

    def clear_all(self):
        """Clear `_literal_cache` + `_builtins_lru` on every live instance (a locked
        snapshot, so a concurrent creation can't perturb the iteration). Best-effort per
        instance — a cache-shape surprise on one must not block the others."""
        with self._lock:
            instances = list(self._all)
        for it in instances:
            try:
                it._literal_cache.clear()
                it._builtins_lru.clear()
            except Exception:
                pass
