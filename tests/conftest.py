"""
Pytest configuration for TEX test suite.

The tests use a shared SubTestResult object (r) to accumulate pass/fail counts.
This fixture provides it and asserts no failures at the end of each test.
"""
import pytest
import torch
from helpers import SubTestResult


@pytest.fixture
def r():
    result = SubTestResult()
    yield result
    if result.failed:
        failures = "\n  ".join(result.errors)
        pytest.fail(f"{result.failed} sub-test(s) failed:\n  {failures}")


@pytest.fixture(autouse=True)
def _race43_stream_leak_guard(request):
    """RACE-43: a CUDA-graph capture that fails partway through (see
    `tex_runtime/graphed.py`'s `capture`/`_recover_from_capture_failure` — a capture-illegal
    sync mid-capture makes `torch.cuda.graph.__exit__`'s `capture_end()` raise, which skips
    the stream restore right after it) can leave THIS THREAD's current CUDA stream pointed
    at a graph-internal capture stream instead of the default one. Nothing else in the
    process ever notices, so the bug surfaces later, in some unrelated test that reads or
    writes tensors through the wrong stream with no fence — a threaded-cache race's rare
    wrong-bytes read, reproduced only inside the whole-suite run and never in isolation.

    This turns that into a NAMED red at the test that causes it. Every test starts on the
    default stream (this fixture resets it on any drift), so if a test still leaves it
    non-default when it returns, that test — not some later, innocent one — is the one
    that fails.

    CUDA-only (a no-op return on a CPU-only box, never a skip: the SIMP-3 skip-site census
    is a source count of `r.skip(...)` sites, and this fixture never calls it). One stream
    compare per test — cheap enough to run in every row of the suite."""
    yield
    if not torch.cuda.is_available():
        return
    current = torch.cuda.current_stream()
    default = torch.cuda.default_stream()
    if current.cuda_stream != default.cuda_stream:
        torch.cuda.set_stream(default)          # never let it cascade into the next test
        pytest.fail(
            f"{request.node.nodeid} left the current CUDA stream non-default "
            f"(was {current.cuda_stream!r}, default is {default.cuda_stream!r}) — reset "
            f"to default so later tests are unaffected")
