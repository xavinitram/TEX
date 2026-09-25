"""OBSERVER-46 — the supported cook-observer seam (`tex_runtime/cook_observer.py`).

ROUTE-45's audit found that a host wrapping `tex_engine.cook_stage_list` to count cooks
misses the calls `tex_chain.cook_fused_cached` makes to ITS OWN `cook_stage_list` and
`boundary_lineage_key` — a re-export plus an internal self-call defeats the wrap. This
seam is the alternative: `register(cb)` is notified once per EXTERNAL cook at each of the
six entry points (`tex_engine.run`/`cook`, `tex_chain.cook_stage_list`/`cook_fused_cached`/
`boundary_lineage_key`, `tex_checkpoint.cook_checkpointed`), collapsing any nesting among
them into the single outermost notification — see `cook_observer.py`'s module docstring
for exactly what "once" means and why.

Every test manages its own registration/teardown with `try`/`finally` rather than a pytest
fixture: `tests/run_all.py`'s canonical runner calls each `test_*(r)` function directly
(`getattr(mod, func_name)(r)`, no pytest fixture machinery), so relying on an autouse
fixture would silently no-op under that runner. Restoring `cook_observer._warned` to
`False` after any test that trips it keeps the once-per-process warning independent of
run order.
"""
import threading

import torch

from TEX_Wrangle import tex_engine, tex_chain, tex_checkpoint, tex_results
from TEX_Wrangle.tex_runtime import cook_observer


def _img():
    return torch.ones(1, 4, 4, 3)


def _two_stage_chain(img):
    """A minimal LINEAR fused chain (CACHE-6 shape): stage 0 reads the source, stage 1
    reads stage 0's handoff via `chain_input`. Valid input for `cook_fused_cached`,
    `cook_checkpointed` and `boundary_lineage_key` alike."""
    return [
        {"code": "@OUT = vec4(@A.rgb * 1.5, 1.0);", "chain_input": None, "bindings": {"A": img}},
        {"code": "@OUT = vec4(@X.rgb + 0.1, 1.0);", "chain_input": "X", "bindings": {}},
    ]


def _register(cb):
    """`register` + return an unregister-on-exit context manager, so every test cleans up
    exactly once even when a body raises partway through."""
    handle = cook_observer.register(cb)

    class _Ctx:
        def __enter__(self):
            return handle

        def __exit__(self, *exc):
            cook_observer.unregister(handle)
            return False

    return _Ctx()


def test_observer46_no_notification_when_nothing_registered(r):
    print("\n--- OBSERVER-46: zero notifications with no callback registered ---")
    try:
        assert not cook_observer._callbacks, "test premise: no callback should be live here"
        img = _img()
        tex_engine.cook("@OUT = @A * 1.5;", {"A": img}, device_mode="cpu")
        plan = tex_engine.prepare("@OUT = @A * 1.5;", {"A": img}, device_mode="cpu")
        tex_engine.run(plan)
        tex_chain.cook_stage_list([{"code": "@OUT = @A * 1.5;", "bindings": {"A": img}}],
                                  device="cpu")
        r.ok("cook/run/cook_stage_list all ran with no registered observer and no error")
    except Exception as e:
        r.fail("OBSERVER-46 no-op", f"{type(e).__name__}: {e}")


def test_observer46_run_and_cook_notify_once_each_as_themselves(r):
    print("\n--- OBSERVER-46: tex_engine.run / tex_engine.cook, each exactly once ---")
    events = []
    with _register(lambda entry, thread: events.append((entry, thread))):
        try:
            img = _img()
            events.clear()
            tex_engine.cook("@OUT = @A * 1.5;", {"A": img}, device_mode="cpu")
            if events == [("cook", threading.current_thread())]:
                r.ok("cook() notified exactly once, as 'cook', on the calling thread")
            else:
                r.fail("OBSERVER-46 cook()", f"expected one ('cook', <this thread>), got {events}")

            events.clear()
            plan = tex_engine.prepare("@OUT = @A * 1.5;", {"A": img}, device_mode="cpu")
            tex_engine.run(plan)
            if events == [("run", threading.current_thread())]:
                r.ok("run(prepare(...)) notified exactly once, as 'run', on the calling thread")
            else:
                r.fail("OBSERVER-46 run()", f"expected one ('run', <this thread>), got {events}")
        except Exception as e:
            r.fail("OBSERVER-46 run/cook", f"{type(e).__name__}: {e}")


def test_observer46_cook_stage_list_notifies_once_directly(r):
    print("\n--- OBSERVER-46: tex_chain.cook_stage_list called directly ---")
    events = []
    with _register(lambda entry, thread: events.append((entry, thread))):
        try:
            img = _img()
            tex_chain.cook_stage_list([{"code": "@OUT = @A * 2.0;", "bindings": {"A": img}}],
                                      device="cpu")
            if events == [("cook_stage_list", threading.current_thread())]:
                r.ok("a direct cook_stage_list() call notified exactly once, as itself")
            else:
                r.fail("OBSERVER-46 cook_stage_list", f"expected one entry, got {events}")
        except Exception as e:
            r.fail("OBSERVER-46 cook_stage_list", f"{type(e).__name__}: {e}")


def test_observer46_boundary_lineage_key_notifies_once_directly(r):
    print("\n--- OBSERVER-46: tex_chain.boundary_lineage_key called directly ---")
    events = []
    with _register(lambda entry, thread: events.append((entry, thread))):
        try:
            stages = _two_stage_chain(_img())
            tex_chain.boundary_lineage_key(stages, 1, "cpu", "fp32", upstream=("srckey",))
            if events == [("boundary_lineage_key", threading.current_thread())]:
                r.ok("a direct boundary_lineage_key() call notified exactly once, as itself")
            else:
                r.fail("OBSERVER-46 boundary_lineage_key", f"expected one entry, got {events}")
        except Exception as e:
            r.fail("OBSERVER-46 boundary_lineage_key", f"{type(e).__name__}: {e}")


def test_observer46_cook_fused_cached_collapses_its_internal_calls(r):
    """The exact ROUTE-45 shape: `cook_fused_cached` calls its OWN `cook_stage_list` (on
    both the cache-miss prefix-materialize path and the suffix cook) and its own
    `boundary_lineage_key` (on the cache-hit path) — none of that must add a second
    notification to the one `cook_fused_cached` itself fires."""
    print("\n--- OBSERVER-46: cook_fused_cached -> its own cook_stage_list/boundary_lineage_key ---")
    events = []
    with _register(lambda entry, thread: events.append(entry)):
        try:
            stages = _two_stage_chain(_img())
            rc = tex_results.ResultCache()

            events.clear()
            tex_chain.cook_fused_cached(stages, 1, rc, device="cpu", upstream=("srckey",))
            if events == ["cook_fused_cached"]:
                r.ok("cache-MISS cook_fused_cached (materializes the prefix) notified once")
            else:
                r.fail("OBSERVER-46 cook_fused_cached miss", f"expected one entry, got {events}")

            events.clear()
            tex_chain.cook_fused_cached(stages, 1, rc, device="cpu", upstream=("srckey",))
            if events == ["cook_fused_cached"]:
                r.ok("cache-HIT cook_fused_cached (boundary_lineage_key + suffix cook) "
                     "notified once, not three times")
            else:
                r.fail("OBSERVER-46 cook_fused_cached hit", f"expected one entry, got {events}")
        except Exception as e:
            r.fail("OBSERVER-46 cook_fused_cached", f"{type(e).__name__}: {e}")


def test_observer46_cook_checkpointed_collapses_its_internal_calls(r):
    """`cook_checkpointed` reaches `cook_stage_list`/`boundary_lineage_key` through
    MODULE-LOOKUP (`tex_engine.cook_stage_list(...)`), the exact shape ROUTE-45 says stays
    covered by a wrap on the old name — and it must stay covered by ONE notification here
    too, whichever of its internal paths it takes."""
    print("\n--- OBSERVER-46: cook_checkpointed -> tex_engine.cook_stage_list/boundary_lineage_key ---")
    events = []
    with _register(lambda entry, thread: events.append(entry)):
        try:
            stages = _two_stage_chain(_img())
            rc = tex_results.ResultCache()

            # Nothing cached yet: _cache_is_provably_empty -> _full() -> one internal
            # cook_stage_list call, which must not add a second notification.
            events.clear()
            tex_checkpoint.cook_checkpointed(stages, rc, device="cpu", upstream=("srckey",),
                                             cuts=[1])
            if events == ["cook_checkpointed"]:
                r.ok("empty-cache cook_checkpointed (whole-chain fallback) notified once")
            else:
                r.fail("OBSERVER-46 cook_checkpointed empty", f"expected one entry, got {events}")

            # Warm the boundary the same way cook_fused_cached's MISS path does, then serve
            # the cut: this walks the per-cut boundary_lineage_key + cook_stage_list loop.
            tex_chain.cook_fused_cached(stages, 1, rc, device="cpu", upstream=("srckey",))
            events.clear()
            tex_checkpoint.cook_checkpointed(stages, rc, device="cpu", upstream=("srckey",),
                                             cuts=[1])
            if events == ["cook_checkpointed"]:
                r.ok("warm-cache cook_checkpointed (boundary probe + suffix cook) notified once")
            else:
                r.fail("OBSERVER-46 cook_checkpointed warm", f"expected one entry, got {events}")
        except Exception as e:
            r.fail("OBSERVER-46 cook_checkpointed", f"{type(e).__name__}: {e}")


def test_observer46_thread_argument_is_the_calling_thread(r):
    print("\n--- OBSERVER-46: the thread passed is the actual calling thread ---")
    seen = {}

    def cb(entry, thread):
        seen[entry] = thread

    with _register(cb):
        try:
            img = _img()
            result = {}

            def _worker():
                tex_engine.cook("@OUT = @A * 1.5;", {"A": img}, device_mode="cpu")
                result["thread"] = threading.current_thread()

            t = threading.Thread(target=_worker, name="observer46-worker")
            t.start()
            t.join(timeout=30)
            if seen.get("cook") is result.get("thread") and seen.get("cook") is t:
                r.ok("the callback saw the worker thread object, not the caller's own")
            else:
                r.fail("OBSERVER-46 thread identity",
                       f"seen={seen.get('cook')!r} worker_self={result.get('thread')!r} "
                       f"started_thread={t!r}")
        except Exception as e:
            r.fail("OBSERVER-46 thread identity", f"{type(e).__name__}: {e}")


def test_observer46_raising_callback_does_not_break_the_cook(r):
    print("\n--- OBSERVER-46: a raising callback never breaks the cook, warns once ---")
    saved_warned = cook_observer._warned
    cook_observer._warned = False
    calls = []

    def bad_cb(entry, thread):
        calls.append(entry)
        raise RuntimeError("observer46 deliberate test failure")

    import warnings
    try:
        with _register(bad_cb):
            img = _img()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                res = tex_engine.cook("@OUT = @A * 1.5;", {"A": img}, device_mode="cpu")
            if res is None or "OUT" not in res.outputs:
                r.fail("OBSERVER-46 raising callback", "cook() did not return a normal result")
            elif calls != ["cook"]:
                r.fail("OBSERVER-46 raising callback", f"callback call log was {calls}")
            elif not any("cook_observer callback raised" in str(w.message) for w in caught):
                r.fail("OBSERVER-46 raising callback", "no warn-once RuntimeWarning was raised")
            else:
                r.ok("a raising callback was caught, warned once, and the cook still returned")

            # Second cook: the callback raises again, but the warn-once latch must not repeat.
            calls.clear()
            with warnings.catch_warnings(record=True) as caught2:
                warnings.simplefilter("always")
                tex_engine.cook("@OUT = @A * 1.5;", {"A": img}, device_mode="cpu")
            if any("cook_observer callback raised" in str(w.message) for w in caught2):
                r.fail("OBSERVER-46 warn-once", "the warning fired a second time in one process")
            else:
                r.ok("the warn-once latch suppressed a second identical warning")
    except Exception as e:
        r.fail("OBSERVER-46 raising callback", f"{type(e).__name__}: {e}")
    finally:
        cook_observer._warned = saved_warned


def test_observer46_register_returns_a_handle_unregister_is_precise(r):
    print("\n--- OBSERVER-46: register/unregister are handle-precise and thread-safe ---")
    a_calls, b_calls = [], []
    ha = cook_observer.register(lambda entry, thread: a_calls.append(entry))
    hb = cook_observer.register(lambda entry, thread: b_calls.append(entry))
    try:
        img = _img()
        tex_engine.cook("@OUT = @A * 1.5;", {"A": img}, device_mode="cpu")
        if a_calls == ["cook"] and b_calls == ["cook"]:
            r.ok("both registered callbacks fired for one external cook")
        else:
            r.fail("OBSERVER-46 register", f"a={a_calls} b={b_calls}")

        cook_observer.unregister(ha)
        a_calls.clear()
        b_calls.clear()
        tex_engine.cook("@OUT = @A * 1.5;", {"A": img}, device_mode="cpu")
        if a_calls == [] and b_calls == ["cook"]:
            r.ok("unregistering one handle silences only that callback")
        else:
            r.fail("OBSERVER-46 unregister", f"a={a_calls} b={b_calls}")

        # A stale/foreign handle is a silent no-op (never raises), so teardown code never
        # needs its own try/except around this call.
        cook_observer.unregister(ha)
        cook_observer.unregister(999999)
        r.ok("re-unregistering a stale handle, and an unknown one, are both silent no-ops")
    except Exception as e:
        r.fail("OBSERVER-46 register/unregister", f"{type(e).__name__}: {e}")
    finally:
        cook_observer.unregister(ha)
        cook_observer.unregister(hb)


def test_observer46_concurrent_register_unregister_is_thread_safe(r):
    print("\n--- OBSERVER-46: concurrent register/unregister from many threads ---")
    errors = []
    handles_lock = threading.Lock()

    def worker(i):
        try:
            for _ in range(200):
                h = cook_observer.register(lambda entry, thread: None)
                cook_observer.unregister(h)
        except Exception as e:                       # pragma: no cover - failure path only
            with handles_lock:
                errors.append(f"worker {i}: {type(e).__name__}: {e}")

    try:
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        if errors:
            r.fail("OBSERVER-46 concurrent register/unregister", "; ".join(errors))
        elif cook_observer._callbacks:
            r.fail("OBSERVER-46 concurrent register/unregister",
                   f"leaked callbacks after teardown: {cook_observer._callbacks}")
        else:
            r.ok("8 threads x 200 register/unregister cycles raced cleanly, nothing leaked")
    except Exception as e:
        r.fail("OBSERVER-46 concurrent register/unregister", f"{type(e).__name__}: {e}")
