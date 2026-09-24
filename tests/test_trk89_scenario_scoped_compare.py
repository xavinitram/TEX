"""TRK-89 — `benchmarks/host_path_counts.py compare()` restricts BOTH legs by `--scenario`.

Before this: `--scenario` narrowed only the CURRENT run (`run_all`'s `only=`). A scenario
added since a baseline was saved arrives on the current side alone; its rows read as
`NEW ROW` and count toward the verdict, so a bare `--compare` against a pre-change baseline
returns rc 1 for a change that moved nothing on any scenario that already existed. This
file pins the fix: `compare(current, baseline_path, scenario=...)` filters the BASELINE'S
flattened rows the same way it always filtered the current run's, so a scoped compare is
scoped on both sides.

Loaded by path, like `helpers.load_counts_harness` already does for this module.
"""
import json
import os

from helpers import load_counts_harness


def _row(total=1, stable=True):
    return {"min": total, "median": total, "max": total, "total": total,
            "stable": stable, "warmup": 0}


def _run(device, scenarios: dict) -> dict:
    return {"device": device, "res": 96, "window": 48, "ticks": 4, "prof1": False,
            "scenarios": scenarios}


def _write_baseline(tmp_path, scenarios: dict) -> str:
    payload = {"env": {"tex_version": "0.0", "tex_sha": "base"},
               "runs": [_run("cpu", scenarios)]}
    p = os.path.join(str(tmp_path), "baseline.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return p


def test_trk89_bare_compare_flags_a_newly_added_scenario(tmp_path, capsys):
    """The pre-existing failure mode: baseline has scenario_a only; current has
    scenario_a (unchanged) + scenario_b (newly added). A bare compare counts
    scenario_b's rows as NEW ROW and returns rc 1 -- even though scenario_a moved
    nothing."""
    b = load_counts_harness()
    baseline_scn = {"scenario_a": {"api": {"row1": _row(1)}, "frames": {}}}
    baseline = _write_baseline(tmp_path, baseline_scn)
    current = {"env": {"tex_version": "0.0", "tex_sha": "head"},
               "runs": [_run("cpu", {
                   "scenario_a": {"api": {"row1": _row(1)}, "frames": {}},
                   "scenario_b": {"api": {"row1": _row(9)}, "frames": {}},
               })]}
    rc = b.compare(current, baseline)
    out = capsys.readouterr().out
    assert rc == 1, "the added scenario's row must not be silently absorbed"
    assert "NEW ROW" in out
    assert "scenario_b" in out


def test_trk89_scenario_filter_restricts_both_legs(tmp_path, capsys):
    """The fix: passing `scenario={'scenario_a'}` excludes scenario_b from BOTH legs,
    so a change that only adds scenario_b (and moves nothing on scenario_a) reads
    clean -- the honest form the tracker row asks for."""
    b = load_counts_harness()
    baseline_scn = {"scenario_a": {"api": {"row1": _row(1)}, "frames": {}}}
    baseline = _write_baseline(tmp_path, baseline_scn)
    current = {"env": {"tex_version": "0.0", "tex_sha": "head"},
               "runs": [_run("cpu", {
                   "scenario_a": {"api": {"row1": _row(1)}, "frames": {}},
                   "scenario_b": {"api": {"row1": _row(9)}, "frames": {}},
               })]}
    rc = b.compare(current, baseline, scenario={"scenario_a"})
    out = capsys.readouterr().out
    assert rc == 0, "scenario_a alone moved nothing; scenario_b must be filtered out"
    assert "scenario_b" not in out
    assert "--scenario filter applied to BOTH legs" in out


def test_trk89_scenario_filter_also_drops_a_scenario_only_the_baseline_has(tmp_path, capsys):
    """Symmetric case: the BASELINE carries a scenario the filtered current run does
    not (because it was run with a narrower `--scenario`). Without the fix that
    scenario's rows would read GONE and fail the verdict; with the filter applied to
    both legs, a scenario outside the requested set never enters the diff at all."""
    b = load_counts_harness()
    baseline_scn = {
        "scenario_a": {"api": {"row1": _row(1)}, "frames": {}},
        "scenario_old": {"api": {"row1": _row(5)}, "frames": {}},
    }
    baseline = _write_baseline(tmp_path, baseline_scn)
    current = {"env": {"tex_version": "0.0", "tex_sha": "head"},
               "runs": [_run("cpu", {"scenario_a": {"api": {"row1": _row(1)}, "frames": {}}})]}
    rc = b.compare(current, baseline, scenario={"scenario_a"})
    out = capsys.readouterr().out
    assert rc == 0
    assert "GONE" not in out
    assert "scenario_old" not in out


def test_trk89_scenario_filter_still_catches_a_real_move_within_the_named_set(tmp_path, capsys):
    """The filter narrows scope; it must not mask a genuine regression on a named
    scenario."""
    b = load_counts_harness()
    baseline = _write_baseline(tmp_path, {"scenario_a": {"api": {"row1": _row(1)}, "frames": {}}})
    current = {"env": {"tex_version": "0.0", "tex_sha": "head"},
               "runs": [_run("cpu", {"scenario_a": {"api": {"row1": _row(2)}, "frames": {}}})]}
    rc = b.compare(current, baseline, scenario={"scenario_a"})
    out = capsys.readouterr().out
    assert rc == 1
    assert "CHANGED" in out
    assert "scenario_a" in out
