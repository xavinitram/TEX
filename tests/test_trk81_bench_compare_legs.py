"""TRK-81 — `benchmarks/eight_config_bench.py --compare` with 3+ legs.

A single `--compare BASELINE` prints one pairing and says nothing about the null
spread that number should be read against (the PERF-7 sitting: 0.923 against one
baseline leg, while a SAME-TREE leg in the same sitting read 1.082 — the 0.923 was
never distinguishable from that noise floor). This file pins the fix: `_pairing_rows`
computes one pairing's per-config geomean/range, `_same_tree` labels a same-tree NULL
control, and `compare_multi` prints every pairing over 3+ legs and can REFUSE (report,
not flag) a sub-threshold geomean when the sitting holds no NULL pairing to calibrate
against.

Loaded by path, like `helpers.load_counts_harness` loads `host_path_counts.py`:
`benchmarks/` is `.comfyignore`d and not a package, so there is no import name.
"""
import importlib.util
import os
import sys

import pytest


def _load_eight_config_bench():
    mod = sys.modules.get("_trk81_eight_config_bench")
    if mod is not None:
        return mod
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(pkg_dir, "benchmarks", "eight_config_bench.py")
    spec = importlib.util.spec_from_file_location("_trk81_eight_config_bench", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_trk81_eight_config_bench"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def b():
    return _load_eight_config_bench()


def _sys(commit, dirty=False):
    return {"git_commit": commit, "git_dirty": dirty}


def _leg(commit, dirty, results):
    return {"system": _sys(commit, dirty), "results": results}


def _results(cfg_vals: dict) -> dict:
    """`{cfg: {prog: median_ms}}` -> the `{cfg: {prog: {"median": ...}}}` shape
    `_pairing_rows`/`_valid_medians` read."""
    return {cfg: {name: {"median": v} for name, v in progs.items()}
            for cfg, progs in cfg_vals.items()}


def test_trk81_same_tree_matches_commit_and_dirtiness(b):
    assert b._same_tree(_sys("abc123"), _sys("abc123")) is True
    assert b._same_tree(_sys("abc123", True), _sys("abc123", True)) is True
    assert b._same_tree(_sys("abc123", True), _sys("abc123", False)) is False, \
        "same commit but different dirtiness is not the same measured tree"
    assert b._same_tree(_sys("abc123"), _sys("def456")) is False
    assert b._same_tree(_sys(None), _sys(None)) is False, \
        "neither side names a commit -- never a null by default"


def test_trk81_pairing_rows_geomean_and_range(b):
    a = _results({"cpu_off_warm": {"p1": 1.0, "p2": 2.0}})
    c = _results({"cpu_off_warm": {"p1": 0.5, "p2": 1.0}})
    rows = b._pairing_rows(a, c)
    s = rows["cpu_off_warm"]
    assert s["n"] == 2
    # a/c per program: 1.0/0.5=2.0, 2.0/1.0=2.0 -> geomean 2.0, range 2.00-2.00
    assert abs(s["geomean"] - 2.0) < 1e-9
    assert abs(s["min"] - 2.0) < 1e-9
    assert abs(s["max"] - 2.0) < 1e-9


def test_trk81_pairing_rows_only_counts_shared_programs(b):
    a = _results({"cpu_off_warm": {"p1": 1.0, "only_a": 3.0}})
    c = _results({"cpu_off_warm": {"p1": 1.0, "only_c": 4.0}})
    s = b._pairing_rows(a, c)["cpu_off_warm"]
    assert s["n"] == 1, "a program missing on one side must not enter the geomean"
    assert abs(s["geomean"] - 1.0) < 1e-9


def test_trk81_compare_multi_labels_null_pairing(b, capsys):
    same = _results({"cpu_off_warm": {"p1": 1.0}})
    legs = [
        ("current", _leg("aaa111", False, same)),
        ("baseline_a", _leg("aaa111", False, same)),   # same tree as current -> NULL
        ("baseline_b", _leg("bbb222", False, _results({"cpu_off_warm": {"p1": 1.0}}))),
    ]
    b.compare_multi(legs, require_null_leg=False)
    out = capsys.readouterr().out
    assert "1 same-tree NULL pairing(s)" in out
    assert "[NULL: same tree]" in out
    assert "current (aaa111) vs baseline_a (aaa111)" in out
    assert "current (aaa111) vs baseline_b (bbb222)" in out
    # the NULL tag decorates only the first pairing's header line, not the second's
    first_block, _, rest = out.partition("baseline_a (aaa111)")
    tagged_line = rest.splitlines()[0]
    assert "[NULL: same tree]" in tagged_line
    second_block = rest.partition("baseline_b (bbb222)")[2].splitlines()[0]
    assert "[NULL: same tree]" not in second_block


def test_trk81_compare_multi_refuses_sub_threshold_without_a_null_leg(b, capsys):
    fast = _results({"cpu_off_warm": {"p1": 1.0}})
    slow = _results({"cpu_off_warm": {"p1": 0.5}})   # current/baseline = 0.5x -> sub-threshold
    legs = [
        ("current", _leg("aaa111", False, slow)),
        ("baseline", _leg("bbb222", False, fast)),      # different tree -- no NULL anywhere
    ]
    b.compare_multi(legs, require_null_leg=True)
    out = capsys.readouterr().out
    assert "NO same-tree NULL leg in this sitting" in out
    assert "REFUSED" in out
    assert "REGRESSION" not in out


def test_trk81_compare_multi_flags_regression_when_a_null_leg_exists(b, capsys):
    same = _results({"cpu_off_warm": {"p1": 1.0}})
    fast = _results({"cpu_off_warm": {"p1": 1.0}})
    slow = _results({"cpu_off_warm": {"p1": 0.5}})
    legs = [
        ("current", _leg("aaa111", False, slow)),
        ("baseline_fast", _leg("bbb222", False, fast)),   # sub-threshold pairing
        ("baseline_null", _leg("aaa111", False, same)),   # NULL: same tree as current
    ]
    b.compare_multi(legs, require_null_leg=True)
    out = capsys.readouterr().out
    assert "1 same-tree NULL pairing(s)" in out
    assert "REGRESSION" in out, \
        "a null leg is present in the sitting, so a real sub-threshold pairing is flagged"
    assert "REFUSED" not in out


def test_trk81_compare_multi_without_require_null_leg_still_flags_regression(b, capsys):
    """`require_null_leg` is opt-in (default False, per the row's 'if the row asks'):
    a plain multi-leg compare with no null leg still reports the numbers and flags a
    sub-threshold geomean, exactly like the existing 2-leg `compare()` always has."""
    fast = _results({"cpu_off_warm": {"p1": 1.0}})
    slow = _results({"cpu_off_warm": {"p1": 0.5}})
    legs = [
        ("current", _leg("aaa111", False, slow)),
        ("baseline_a", _leg("bbb222", False, fast)),
        ("baseline_b", _leg("ccc333", False, fast)),
    ]
    b.compare_multi(legs, require_null_leg=False)
    out = capsys.readouterr().out
    assert "REGRESSION" in out
    assert "REFUSED" not in out
