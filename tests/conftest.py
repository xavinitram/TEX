"""
Pytest configuration for TEX test suite.

The tests use a shared TestResult object (r) to accumulate pass/fail counts.
This fixture provides it and asserts no failures at the end of each test.
"""
import pytest
from test_tex import TestResult


@pytest.fixture
def r():
    result = TestResult()
    yield result
    if result.failed:
        failures = "\n  ".join(result.errors)
        pytest.fail(f"{result.failed} sub-test(s) failed:\n  {failures}")
