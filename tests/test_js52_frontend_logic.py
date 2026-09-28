"""Behavioural checks for the pure logic in `js/tex_extension.js`.

The ComfyUI extension is an ES module that imports the host's scripts, so its parsing and
socket-sync helpers are exercised by `tests/js_frontend_checks.cjs`, which cuts them out by
name and runs them under plain node against a fake graph. This wrapper runs that script and
fails with its output. Without node the checks cannot run at all, so the test is skipped.
"""
import pathlib
import shutil
import subprocess

import pytest

_TESTS = pathlib.Path(__file__).resolve().parent


def test_frontend_logic_under_node():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    proc = subprocess.run([node, str(_TESTS / "js_frontend_checks.cjs")],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_extension_parses():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    proc = subprocess.run([node, "--check", str(_TESTS.parent / "js" / "tex_extension.js")],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
