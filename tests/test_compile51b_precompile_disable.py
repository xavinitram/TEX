"""
COMPILE-51b (v0.51, host item 2's own "or say precisely why not") — the `caching_precompile`
guard-state pickling crash filed as the COMPILE-51 finding.

BEFORE this ask: `compiled._precompile_ctx()` unconditionally scoped dynamo's
`caching_precompile` ON around the ENTIRE compile-and-first-invoke window (both
`execute_compiled`'s `_compile_and_run` and `_submit_bg_compile`'s
`_compile_and_maybe_warm`). Any program in the `_has_fn_calls` class `fncalls_compile`
already tracks (COMPILETRY-50) -- `gauss_blur`/`bilateral_filter`, both of which reach
`stdlib_core._get_gauss_kernels`'s own `@torch._dynamo.disable()`-guarded kernel-cache
lookup -- hit, on the FIRST real `torch.compile` attempt for that fingerprint:
`torch._dynamo.exc.InternalTorchDynamoError: TypeError: cannot pickle '_thread._local'
object`, raised from `torch/_dynamo/guards.py:pickle_guards_state` while serializing the
RESUME frame's guard state for the continuation right after the disabled call.
`compiled.py`'s own never-hard-fail net caught it and PERMANENTLY blacklisted the
fingerprint (the verdict survives a process restart via `warm_state.json`) -- silently
demoting to codegen-only/interpreter forever, even though the identical program compiles
and runs correctly once `caching_precompile` is off. Confirmed present at base `707502a`,
unmodified, on real CUDA hardware (RTX 5070 Ti laptop, torch 2.12+cu130): directly, and via
the pre-existing `tests/test_fuseddev46_device.py`'s two CUDA-gated rows, which already
errored this exact way before this ask's fix landed.

AFTER: `_precompile_ctx(disable=True)` scopes `caching_precompile` OFF instead of ON --
in-memory-only `torch.compile()`, no disk persistence -- for exactly the narrow
`_has_fn_calls` class (`_wants_precompile_off`, probed once per call via
`_get_or_make_codegen_fn`'s own per-fingerprint memo, PC-3 -- no extra emit cost). Every
other compiled program (the common, no-stdlib-call case) keeps full disk-persisted
`caching_precompile` unchanged, so PC-2's warm-restart win is not forfeited generally.
`_precompile_flag_lock` serializes the two cases against each other because the dynamo
config flag this toggles is PROCESS-GLOBAL while `_COMPILE_POOL` and `_WARM_POOL` are two
independent single-worker pools that can run genuinely concurrently
(`fncalls_compile.py`'s own header names this exact hazard) -- without it, one pool's
off-scoped attempt could flip the flag under an unrelated concurrent on-scoped compile on
the other pool.

WHAT K1 (FIX-COMPILE, v0.50 Phase C) STILL NEEDS with `caching_precompile` off for this
class: K1's per-build synthetic `sys.modules` entry (`codegen_persist._codegen_exec_namespace`)
fixes a SEPARATE, unconditional crash (`KeyError: '__name__'` in codegen's exec namespace,
reached by ANY graph break needing a Dynamo resume, `caching_precompile` on OR off) -- it is
not scoped to `caching_precompile` at all, so turning `caching_precompile` off for the
`_has_fn_calls` class does not make K1's module dispensable; a resume frame in this class
still needs a real, registered module to resolve `__name__` against. The two fixes are
orthogonal and both required.

PORTABILITY. The mechanism (`_precompile_ctx`'s new `disable=` branch, `_wants_precompile_off`'s
probe, `_precompile_flag_lock`'s serialization) is proven here with CPU-only stand-ins --
monkeypatched `torch._dynamo.config` and a fake `cg_fn`, mirroring
`test_compiletry50_fncalls_gate.py`'s own CC-5 rationale ("the mechanism under test... does
not need a real backend to prove"). The actual pickling crash needs real CUDA/Triton/Inductor
and a real dynamo resume frame to reproduce; `tests/test_fuseddev46_device.py`'s two CUDA-gated
rows are the real-hardware proof (this ask updated their tolerance from bit-exact `torch.equal`
to invariant 2's own 1e-5, because unblocking the real backend surfaces a real, tiny,
in-tolerance Inductor kernel-fusion reassociation that a crash-then-fallback previously hid).
"""
import threading
import time

import pytest

from helpers import *  # noqa: F401,F403
from TEX_Wrangle.tex_runtime import compiled as C
from TEX_Wrangle.tex_runtime import compiled_precompile as CP
# C2 (R1): reuse the byte-identical `_fake_cg_fn` this file used to carry its own copy of,
# rather than a second definition of the same stand-in.
from test_compiletry50_fncalls_gate import _fake_cg_fn  # noqa: F401 (re-used below)

dynamo_config = pytest.importorskip("torch._dynamo.config")
# C2 (B4#5): three tests below set `dynamo_config.caching_precompile` directly with no
# check that the attribute exists on the installed torch -- on a build without this
# flag, a validated `ConfigModule` raises `AttributeError` for an unrecognized key
# rather than silently creating one, so those tests would ERROR rather than SKIP.
# Guard once, at module level, instead of repeating the check in every test.
pytestmark = pytest.mark.skipif(not hasattr(dynamo_config, "caching_precompile"),
                                reason="this torch build has no caching_precompile flag to scope")


@pytest.fixture(autouse=True)
def _restore_flag():
    """Never let a test leak a flipped `caching_precompile` into a later test."""
    had = getattr(dynamo_config, "caching_precompile", None)
    yield
    if had is not None:
        dynamo_config.caching_precompile = had


def test_wants_precompile_off_true_for_fn_calls_class(monkeypatch):
    """`_wants_precompile_off` reads `cg_fn._has_fn_calls` via
    `_get_or_make_codegen_fn`'s own memo -- no real codegen pipeline needed."""
    monkeypatch.setattr(C, "_get_or_make_codegen_fn",
                        lambda program, type_map, fp: _fake_cg_fn(has_fn_calls=True))
    assert C._wants_precompile_off(object(), {"x": 1}, "fp-a") is True


def test_wants_precompile_off_false_for_ordinary_program(monkeypatch):
    monkeypatch.setattr(C, "_get_or_make_codegen_fn",
                        lambda program, type_map, fp: _fake_cg_fn(has_fn_calls=False))
    assert C._wants_precompile_off(object(), {"x": 1}, "fp-b") is False


def test_wants_precompile_off_false_when_program_or_type_map_missing():
    """No program/type_map to probe (e.g. the interpreter-wrapping path never reached
    for real) -- never crash the probe, and never claim the fn-calls class."""
    assert C._wants_precompile_off(None, {"x": 1}, "fp-c") is False
    assert C._wants_precompile_off(object(), None, "fp-c") is False


def test_wants_precompile_off_false_on_probe_exception(monkeypatch):
    """A probe failure (e.g. codegen raises on this program) must fail CLOSED — never
    claim the fn-calls class from an exception, which would silently and permanently
    forfeit `caching_precompile`'s disk persistence for a program that never earns it."""
    def _boom(program, type_map, fp):
        raise RuntimeError("emit failed")
    monkeypatch.setattr(C, "_get_or_make_codegen_fn", _boom)
    assert C._wants_precompile_off(object(), {"x": 1}, "fp-d") is False


def test_precompile_ctx_disable_scopes_flag_off_and_restores():
    """RED at base `707502a`: `_precompile_ctx()` took no `disable=` argument at all, so
    this call raised `TypeError` before ever reaching the flag. AFTER: `disable=True` sets
    `caching_precompile` False for the scope, and the pre-existing value (True, set up by
    the outer harness below to mimic the always-on-before-this-ask default) is restored on
    exit -- proving the fix does not leak the disabled state into the NEXT program's own
    (default) on-scoped compile."""
    dynamo_config.caching_precompile = True
    with C._precompile_ctx(disable=True):
        assert dynamo_config.caching_precompile is False
    assert dynamo_config.caching_precompile is True


def test_precompile_ctx_default_still_scopes_flag_on():
    """The unaffected, common (no-stdlib-call) case: `disable=False` (the default) keeps
    today's behaviour unchanged -- `caching_precompile` ON for the scope, so PC-2's
    warm-restart win is not forfeited for programs outside the `_has_fn_calls` class."""
    dynamo_config.caching_precompile = False
    with C._precompile_ctx():
        assert dynamo_config.caching_precompile is True
    assert dynamo_config.caching_precompile is False


def test_precompile_flag_lock_serializes_two_off_scoped_callers(monkeypatch):
    """`_precompile_flag_lock` must be held across the WHOLE off-scoped window, not just
    the flag flip -- two threads both requesting `disable=True` must never observe each
    other's window overlapping (the process-global flag would otherwise be flipped back ON
    by whichever thread exits first, corrupting the OTHER thread's still-running compile).
    Forces the overlap with a barrier-like sleep inside the `with` block and records
    enter/exit order. Forces the shared-global build shape: a per-thread patch needs no lock."""
    monkeypatch.setattr(CP, "_patch_thread_local", False)
    dynamo_config.caching_precompile = True
    events = []
    lock = threading.Lock()

    def _worker(tag):
        with C._precompile_ctx(disable=True):
            with lock:
                events.append((tag, "enter"))
            # Give the OTHER thread a window to (wrongly) start its own off-scope
            # while this one is still inside, if the lock did not actually serialize.
            import time as _t
            _t.sleep(0.05)
            with lock:
                events.append((tag, "exit"))

    t1 = threading.Thread(target=_worker, args=("A",))
    t2 = threading.Thread(target=_worker, args=("B",))
    t1.start()
    t2.start()
    t1.join(5)
    t2.join(5)

    assert len(events) == 4
    # Whichever thread entered first must also exit before the other enters --
    # i.e. no interleaving of the form [A-enter, B-enter, A-exit, B-exit].
    first_tag = events[0][0]
    assert events[1] == (first_tag, "exit"), (
        f"off-scoped windows overlapped, lock did not serialize: {events}")
    assert dynamo_config.caching_precompile is True


@pytest.mark.timing
def test_precompile_ctx_default_branch_also_blocks_on_the_shared_lock(monkeypatch):
    """FIX-COMPILE51 C1 (B3#1): RED at `365fdb4` -- `_precompile_flag_lock` was taken only
    by the `disable=True` (OFF-scoped) branch (`_precompile_off_ctx`); the `disable=False`
    (default, ON-scoped) branch called `_dc.patch(caching_precompile=True)` directly, never
    touching the lock at all. That means an off-scoped compile on one pool (`_COMPILE_POOL`/
    `_WARM_POOL`) and a concurrent default-scoped compile on the other were never actually
    serialized against each other, despite the module comment's and this file's own
    `test_precompile_flag_lock_serializes_two_off_scoped_callers` claiming exactly that
    coverage -- that test only ever pits two `disable=True` callers against each other, so
    it cannot catch the missing lock in the `disable=False` branch.

    This proves the missing coverage directly, on the LOCK OBJECT itself, rather than by
    sampling `caching_precompile`'s live value across threads: this torch build's
    `ConfigModule.patch()` stores the patched value in a `contextvars.ContextVar`, which
    Python isolates per OS thread by design (confirmed separately: a value one thread
    `.set()`s is invisible to a concurrently running thread's own `.get()`), so two real
    `threading.Thread`s sampling the flag value across each other's windows would read
    each thread's own default and never observe the cross-thread symptom B3#1 measured on
    its own box -- a test built that way would be RED for the wrong reason at best, and a
    false pass at worst (exactly the class of weak test B4#1 flagged elsewhere in this
    diff). The lock is the actual shared, cross-thread-visible resource FIX-COMPILE51 must
    cover for both branches, so this test holds `_precompile_flag_lock` externally
    (simulating a live off-scoped compile in progress on the OTHER pool) and asserts that
    entering the default (`disable=False`) scope BLOCKS until it is released -- AFTER the
    fix, both branches share one lock-wrapped helper, so entering either one while the
    other's window is open must wait for it (on a build whose patch is a shared global; a
    per-thread patch is covered by the test below)."""
    monkeypatch.setattr(CP, "_patch_thread_local", False)
    dynamo_config.caching_precompile = False

    def _holder():
        with C._precompile_flag_lock:
            time.sleep(0.1)  # hold for a fixed window, released on its own

    holder = threading.Thread(target=_holder)
    holder.start()
    try:
        time.sleep(0.02)  # let the holder actually acquire the lock first
        start = time.monotonic()
        ctx = C._precompile_ctx(disable=False)
        ctx.__enter__()
        try:
            elapsed = time.monotonic() - start
        finally:
            ctx.__exit__(None, None, None)
    finally:
        holder.join(5)

    assert elapsed >= 0.04, (
        f"entering the default (disable=False) scope returned in {elapsed * 1000:.1f}ms "
        f"while _precompile_flag_lock was held by another caller -- the disable=False "
        f"branch does not actually acquire the shared lock")


@pytest.mark.timing
def test_a_per_thread_patch_does_not_wait_for_another_pools_compile(monkeypatch):
    """On a build whose config patch is per-thread (torch 2.12), one pool's long compile
    must not hold up the other pool's scope entry."""
    monkeypatch.setattr(CP, "_patch_thread_local", None)
    if not CP._patch_is_thread_local(dynamo_config):
        return   # a shared-global build keeps the lock; the two tests above cover that shape
    holder_in = threading.Event()
    release = threading.Event()

    def _long_compile():
        with C._precompile_ctx(disable=False):
            holder_in.set()
            release.wait(5)

    holder = threading.Thread(target=_long_compile)
    holder.start()
    try:
        assert holder_in.wait(5)
        start = time.monotonic()
        with C._precompile_ctx(disable=True):
            elapsed = time.monotonic() - start
    finally:
        release.set()
        holder.join(5)
    assert elapsed < 0.5, f"second scope waited {elapsed * 1000:.0f}ms behind the first"
