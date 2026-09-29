"""Promoted noise in a long single process: a compile-tier failure is never a cook error.

A tiered noise key is promoted to `torch.compile` after a few calls. Dynamo keeps its compiled
specializations per CODE OBJECT and stops at `recompile_limit` (8) of them; with
`fullgraph=True` the next one raises instead of running eagerly. Every
`_make_fbm_fast_fn(octaves)` closure shared one `fbm_fn` code object, and each promotion's
warm-up passed the same tensor as x and y (a separate "duplicate tensors" specialization), so
a handful of octave counts filled the budget and the next new signature (a batch of 2, a
transposed view) raised out of the cook. The same exit was open to any other error the
promoted tier raises at a new signature, such as a C++ build failure.

Box-independent: the promotion is forced (`_inductor_available`) and compiled with Dynamo's
`eager` backend, so no C++ compiler or Triton is needed and the guards and the recompile limit
are Dynamo's real ones. Each test uses fresh caches so no earlier test's state leaks in.
"""
import pytest
import torch

from TEX_Wrangle.tex_runtime import noise, tier_trace


@pytest.fixture
def fresh_fbm(monkeypatch):
    """A fresh fbm cache, promotion forced on for CPU, Dynamo's eager backend."""
    cache = noise._TieredCache("fbm")
    monkeypatch.setattr(noise, "_fbm_cache", cache)
    monkeypatch.setitem(noise._inductor_available, "cpu", True)
    monkeypatch.setattr(noise, "_NOISE_COMPILE_BACKEND", "eager")
    return cache


def _promote(x, y, octaves):
    for _ in range(noise._COMPILE_AFTER_CALLS + 1):
        noise._fbm2d(x, y, octaves)


def _tier(cache, key):
    fn = cache.cache.get(key)
    if fn is None or fn is False:
        return "eager"
    return "trace" if isinstance(fn, torch.jit.ScriptFunction) else "promoted"


def test_many_octave_counts_and_new_signatures_never_raise(fresh_fbm):
    """The red row: four promoted octave counts, then new signatures on each. On the shared
    code object this raised FailOnRecompileLimitHit out of `_fbm2d`."""
    torch.manual_seed(3)
    x, y = torch.rand(1, 24, 32) * 8, torch.rand(1, 24, 32) * 8
    octaves = (1, 2, 3, 4)
    for o in octaves:
        _promote(x, y, o)
    new_sigs = [
        (torch.rand(2, 24, 32) * 8, torch.rand(2, 24, 32) * 8),       # batch 1 -> 2
        (x.transpose(1, 2), y.transpose(1, 2)),                       # new strides
        (torch.rand(5, 6) * 8, torch.rand(5, 6) * 8),                 # new rank
    ]
    for o in octaves:
        for a, b in new_sigs:
            got = noise._fbm2d(a, b, o)
            want = noise._make_fbm_fast_fn(o)(a, b)
            assert torch.allclose(got, want, atol=1e-5), (o, tuple(a.shape))
    # Every promotion has its own specialization budget and the warm-up's specialization is
    # the one real calls reuse, so nothing above needed to fall back.
    assert [_tier(fresh_fbm, (o, x.device)) for o in octaves] == ["promoted"] * 4


def test_recompile_limit_hit_falls_back_to_the_trace_tier(fresh_fbm):
    """Past the limit the key goes back to its jit.trace tier, answers correctly, and the
    failure is recorded (not swallowed, not raised into the cook)."""
    torch.manual_seed(4)
    x, y = torch.rand(1, 24, 32) * 8, torch.rand(1, 24, 32) * 8
    _promote(x, y, 5)
    key = (5, x.device)
    assert _tier(fresh_fbm, key) == "promoted"
    before = len(tier_trace.noise_compile_failures())
    a, b = torch.rand(3, 24, 32) * 8, torch.rand(3, 24, 32) * 8
    with torch._dynamo.config.patch(recompile_limit=1):
        got = noise._fbm2d(a, b, 5)
    assert torch.allclose(got, noise._make_fbm_fast_fn(5)(a, b), atol=1e-5)
    assert _tier(fresh_fbm, key) == "trace"
    failures = tier_trace.noise_compile_failures()
    assert len(failures) == before + 1 and "RecompileLimit" in failures[-1]["error"]
    # The next call stays on the trace tier and still answers.
    assert torch.allclose(noise._fbm2d(a, b, 5), got, atol=1e-5)
    assert noise._inductor_available["cpu"] is True     # a per-key fact, not process-wide


def _cpp_compile_error():
    try:
        from torch._inductor.exc import CppCompileError
        return CppCompileError(["cl"], "C1083: cannot open compiler generated file")
    except Exception:
        return RuntimeError("C++ compile error")


def test_a_promoted_tier_build_failure_answers_from_the_trace():
    """A promoted callable that raises at a new signature (the whole-suite CppCompileError)
    is demoted to the trace it replaced; the cook gets the right value."""
    cache = noise._TieredCache("probe-promoted-raises")
    trace_calls = []

    def trace(a):
        trace_calls.append(1)
        return a * 2.0

    def promoted(a):
        raise _cpp_compile_error()

    cache.cache["k"] = trace
    cache._fallback["k"] = trace
    cache.cache["k"] = promoted
    got = cache._settle("k", promoted, lambda a: a * 2.0, (torch.ones(3),))
    assert torch.equal(got, torch.full((3,), 2.0))
    assert cache.cache["k"] is trace and trace_calls


def test_a_failing_tier_with_no_trace_behind_it_demotes_to_eager():
    cache = noise._TieredCache("probe-trace-raises")

    def boom(a):
        raise RuntimeError("a tier-only failure")

    cache.cache["k"] = boom
    got = cache._settle("k", boom, lambda a: a * 4.0, (torch.ones(3),))
    assert torch.equal(got, torch.full((3,), 4.0))
    assert cache.cache["k"] is False


def test_an_error_eager_raises_too_surfaces_and_leaves_the_key_alone():
    """A genuine error (every tier raises it, the eager oracle included) still reaches the
    cook, and the tier that raised it is not demoted for it."""
    cache = noise._TieredCache("probe-genuine")

    def boom(a):
        raise RuntimeError("a genuine error")

    def eager(a):
        raise RuntimeError("a genuine error")

    cache.cache["k"] = boom
    with pytest.raises(RuntimeError, match="genuine"):
        cache._settle("k", boom, eager, (torch.ones(3),))
    assert cache.cache["k"] is boom


def test_out_of_memory_is_never_turned_into_a_fallback():
    """OOM propagates untouched: the engine's own OOM handling needs to see it."""
    cache = noise._TieredCache("probe-oom")

    def oom(a):
        raise torch.OutOfMemoryError("CUDA out of memory")

    cache.cache["k"] = oom
    with pytest.raises(torch.OutOfMemoryError):
        cache._settle("k", oom, lambda a: a, (torch.ones(3),))
    assert cache.cache["k"] is oom
