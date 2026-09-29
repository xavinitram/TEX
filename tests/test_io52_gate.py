"""
The gate's cache and junit handling: a verdict is keyed on everything it was measured under, an
environment fault is not cached, and a reused scratch directory cannot feed a stale report.
Stdlib subprocesses stand in for pytest; no real suite is collected.
"""
import importlib.util
import os
import sys
import tempfile

import pytest


def _gate():
    mod = sys.modules.get("_io52_gate")
    if mod is not None:
        return mod
    import pathlib
    path = pathlib.Path(__file__).resolve().parent.parent / "tools" / "gate.py"
    spec = importlib.util.spec_from_file_location("_io52_gate", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_io52_gate"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_tree_hash_without_git_never_repeats(monkeypatch):
    g = _gate()
    monkeypatch.setattr(g, "enumerate_paths", lambda cwd: None)
    assert g.tree_hash() != g.tree_hash()


def test_tree_hash_is_stable_when_git_answers(monkeypatch):
    g = _gate()
    monkeypatch.setattr(g, "enumerate_paths", lambda cwd: ["tools/gate.py"])
    assert g.tree_hash() == g.tree_hash()


def test_cache_key_follows_the_counts_baseline_contents():
    g = _gate()
    ifc = [("python", sys.executable)]
    a = g.cache_key("t", "full", ifc, True, False, "aaa")
    assert a != g.cache_key("t", "full", ifc, True, False, "bbb")
    assert a == g.cache_key("t", "full", ifc, True, False, "aaa")


def test_file_digest_changes_with_the_file():
    g = _gate()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "base.json")
        with open(p, "w") as f:
            f.write("one")
        first = g._file_digest(p)
        with open(p, "w") as f:
            f.write("two")
        assert g._file_digest(p) != first
    assert g._file_digest(None) == ""


_STALE = ('<testsuites><testsuite tests="1"><testcase classname="tests.test_x" name="test_ok"/>'
          '</testsuite></testsuites>')


def test_a_stale_junit_report_is_not_read_as_this_runs_result():
    g = _gate()
    with tempfile.TemporaryDirectory() as scratch:
        leg = g.Leg("stale", "a leg that writes no report of its own")
        with open(os.path.join(scratch, "junit-stale.xml"), "w") as f:
            f.write(_STALE)
        g._run(leg, [sys.executable, "-c", "pass"], scratch, {}, scratch, False)
        assert leg.failures == ["<stale:infra-rc0>"]
        assert g.judge([leg], [], False)["verdict"] == "RED"


def test_infra_red_is_recognised():
    g = _gate()
    bad, ok = g.Leg("a", ""), g.Leg("b", "")
    bad.failures = ["<a:infra-rc1>"]
    ok.failures = ["tests/test_x.py::test_y"]
    assert g._has_infra_red([bad]) and not g._has_infra_red([ok]) and not g._has_infra_red([])


def _drive_main(monkeypatch, leg):
    g = _gate()
    written = []
    monkeypatch.setattr(g, "run_cheap", lambda *a, **k: leg)
    monkeypatch.setattr(g, "tree_hash", lambda: "T" * 64)
    monkeypatch.setattr(g, "head_label", lambda: "abc")
    monkeypatch.setattr(g, "_cache_read", lambda key: None)
    monkeypatch.setattr(g, "_cache_write", lambda key, rec: written.append(rec))
    monkeypatch.setattr(g, "load_allowlist", lambda: [])
    monkeypatch.setattr(g, "_prune_inductor_cache_root", lambda *a, **k: None)
    monkeypatch.setattr(g, "interpreter_identity", lambda path: (path, "3"))
    monkeypatch.setattr(g, "_importable_as_tex_wrangle", lambda: True)
    with tempfile.TemporaryDirectory() as scratch:
        code = g.main(["--tier", "cheap", "--scratch", scratch])
    return code, written


def test_an_environment_fault_is_not_cached(monkeypatch):
    g = _gate()
    leg = g.Leg("cheap", "stub")
    leg.rc, leg.failures, leg.collected = 1, ["<cheap:infra-rc1>"], set()
    code, written = _drive_main(monkeypatch, leg)
    assert code == 1 and written == []


def test_a_real_verdict_is_still_cached(monkeypatch):
    g = _gate()
    leg = g.Leg("cheap", "stub")
    leg.rc, leg.failures, leg.collected = 0, [], {"tests/test_x.py::test_y"}
    code, written = _drive_main(monkeypatch, leg)
    assert code == 0 and len(written) == 1 and written[0]["verdict"] == "GREEN"


def test_touched_module_maps_package_init_and_root_files():
    g = _gate()
    assert g._touched_module("__init__.py") is None
    assert g._touched_module("tex_engine.py") == "tex_engine"
    assert g._touched_module("tex_runtime/compiled.py") == "tex_runtime.compiled"
