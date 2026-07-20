"""
TEX Wrangle — the engine session (DATA-4, phase 1).

The engine's cook state is spread across module-level singletons: the program cache
(`tex_cache.get_cache`), the cache governor (`tex_memory.get_cache_registry`), the host
services (`tex_runtime.host.get_host_services`), and the per-thread interpreter pool
(`tex_engine._get_interpreter`). That is fine for one process running one host, but a
compositor runs for days and a standalone host wants ONE handle to hold, inspect, and reset
that state — not six imports and six reset functions.

`EngineSession` is that handle. **Phase 1** (this): there is exactly ONE session, the process
default, and its `.cache` / `.registry` / `.host` / `.interpreter` ARE the module-level
singletons (views, not copies) — so ComfyUI is **byte-identical**, and a test or host that
still reaches a module global sees the same object. The value it adds today is a single
lifecycle surface (`reset()` / `close()` / `stats()`) over state that otherwise lives in six
places, plus the soak-test seam that guards against slow leaks (`tests/test_v028_phase1`).

**Phase 2** (deferred, follows ENG-1): threading a session THROUGH `engine.cook(session=...)`
so a second, ISOLATED session can own its own caches — that is the real multi-tenant story and
needs the cook signature change ENG-1 set up. Until then, creating more than one session hands
back views of the same default; `isolated=False` says so.

THREAD-SAFETY (the written contract, extending DEVELOPMENT.md's ENG-9 section): a session does
NOT add a lock. The default session's objects are the same single-cook-thread singletons ENG-9
classified; `reset()` runs the existing `free_tensor_caches()` sweep, which is safe on the one
cook thread and unsafe to call concurrently with a live cook (same as calling it directly
today). A parallel executor (GRAPH-2) that wants per-thread sessions is phase 2 + the MUT-cache
sharding ENG-9 already flags — a session handle does not change that boundary, it names it.
"""
from __future__ import annotations


class EngineSession:
    """One handle over the process's cook state (DATA-4). Phase 1: a view of the module-level
    singletons — see the module docstring. Construct via `default_session()`; a bare
    `EngineSession()` is still a view of the same globals (`isolated` is False) until phase 2."""

    isolated = False   # phase 2: an isolated session owns its own caches (threaded through cook)

    @property
    def cache(self):
        """The program / codegen compile cache (`tex_cache.TEXCache`)."""
        from .tex_cache import get_cache
        return get_cache()

    @property
    def registry(self):
        """The CACHE-5 cache governor (`tex_memory.CacheRegistry`) arbitrating the VRAM pools."""
        from .tex_memory import get_cache_registry
        return get_cache_registry()

    @property
    def host(self):
        """The host services (`ComfyHostServices` under ComfyUI, else `NullHostServices`)."""
        from .tex_runtime.host import get_host_services
        return get_host_services()

    @property
    def interpreter(self):
        """This thread's interpreter (the ENG-9 per-thread pool). A different cook thread gets a
        different instance — that is deliberate (per-instance mutable execution state)."""
        from .tex_engine import _get_interpreter
        return _get_interpreter()

    def set_host(self, services) -> None:
        """Install host services (e.g. a standalone host's `NullHostServices`) for this process."""
        from .tex_runtime.host import set_host_services
        set_host_services(services)

    def stats(self, device_type: str = "cuda") -> dict:
        """Per-pool cache bytes on `device_type`, via the governor (stdlib / graphs / frames)."""
        return self.registry.stats(device_type)

    def reset(self) -> None:
        """Shed all in-memory tensor caches — the state a long-running host drops between
        projects (the umbrella `free_tensor_caches`: stdlib pyramids, noise, CUDA graphs via
        `free_graphs_only`, and every thread's interpreter caches). Program/codegen DISK caches
        PERSIST (that is the point of a warm relaunch). Single-cook-thread; never call it
        concurrently with a live cook."""
        from .tex_memory import free_tensor_caches
        free_tensor_caches()

    def close(self) -> None:
        """Shut the session down: `reset()` the tensor caches, then drop host services back to
        the process default. A standalone host calls this at exit (engine-era ENG-11 grows this
        into a full lifecycle)."""
        self.reset()
        from .tex_runtime.host import reset_host_services
        reset_host_services()


_default: EngineSession | None = None


def default_session() -> EngineSession:
    """The process's single default session (DATA-4 phase 1). Lazily created; its caches ARE the
    module-level singletons, so holding it changes nothing about how ComfyUI cooks."""
    global _default
    if _default is None:
        _default = EngineSession()
    return _default
