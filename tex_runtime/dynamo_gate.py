"""Keeps `torch._dynamo.reset()` away from code that is compiling or running under Dynamo.

Dynamo's caches are process-global. A reset frees every compiled entry, its guards and its
resume functions, so a reset while another thread is inside a Dynamo compile or a compiled
frame is a use-after-free in native code. That kills the whole host process (seen as
SIGFPE on Linux and as an access violation on Windows), and no Python handler can catch it.

Every thread that runs Dynamo work does it inside `dynamo_job()`. TEX resets Dynamo only
through `reset_if_idle()`, which skips the reset while any job is inside. A skipped reset
costs memory until the next idle one; it never costs correctness, since a failed
artifact is already dropped from TEX's own caches.

A pure leaf: no import of any sibling `tex_runtime` module."""
from __future__ import annotations

import contextlib
import threading

_lock = threading.Lock()
_active = 0


@contextlib.contextmanager
def dynamo_job():
    """Marks the current thread as inside Dynamo work for the duration of the block."""
    global _active
    with _lock:
        _active += 1
    try:
        yield
    finally:
        with _lock:
            _active -= 1


def reset_if_idle() -> bool:
    """`torch._dynamo.reset()`, unless a `dynamo_job()` is running anywhere in the process.
    Returns True when the reset ran. The lock is held across the reset, so no job can start
    halfway through it."""
    with _lock:
        if _active:
            return False
        import torch
        torch._dynamo.reset()
    return True


def active_jobs() -> int:
    """How many `dynamo_job()` blocks are running now."""
    return _active
