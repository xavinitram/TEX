"""TRK-100 — `benchmarks/host_path_counts.py environment()` records the measurement
SHAPE, and `--compare` refuses a mismatched pair.

Before this: `environment()` recorded the version, sha, cache dir and cache warmth
(TRK-45) but not `--res`/`--window`/`--ticks`/`--device` — the shape a saved baseline's
counts are coupled to. Comparing two saves taken at different shapes silently reported
every row as `NEW ROW`/`GONE` (every row genuinely differs structurally at a different
window/resolution) instead of saying the SHAPES differ, which is a much clearer and much
cheaper diagnosis. This file pins both halves: `environment()` carries the four fields,
and `compare()` refuses outright (prints the mismatched field(s) by name, returns rc 1,
and does not run the row diff at all) when they disagree.

Loaded by path, like `helpers.load_counts_harness` already does for this module.
"""
import json
import os

from helpers import load_counts_harness


def _row(total=1, stable=True):
    return {"min": total, "median": total, "max": total, "total": total,
            "stable": stable, "warmup": 0}


def _run(device, res, window, ticks, scenarios: dict) -> dict:
    return {"device": device, "res": res, "window": window, "ticks": ticks,
            "prof1": False, "scenarios": scenarios}


def _write(tmp_path, name, env: dict, runs: list) -> str:
    p = os.path.join(str(tmp_path), name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"env": env, "runs": runs}, f)
    return p


def test_trk100_environment_carries_the_measurement_shape():
    b = load_counts_harness()
    env = b.environment(res=96, window=48, ticks=4, device="cpu")
    assert env["res"] == 96
    assert env["window"] == 48
    assert env["ticks"] == 4
    assert env["device"] == "cpu"
    # the pre-existing fields (TRK-45) must still be present
    assert "tex_cache_dir" in env
    assert "tex_cache_warmth" in env
    assert "tex_sha" in env


def test_trk100_environment_defaults_are_none_when_unasked(tmp_path):
    b = load_counts_harness()
    env = b.environment()
    assert env["res"] is None and env["window"] is None
    assert env["ticks"] is None and env["device"] is None


def test_trk100_compare_refuses_a_resolution_mismatch(tmp_path, capsys):
    b = load_counts_harness()
    scn = {"scenario_a": {"api": {"row1": _row(1)}, "frames": {}}}
    baseline_env = {"tex_version": "0.0", "tex_sha": "base",
                    "res": 1024, "window": 512, "ticks": 8, "device": "cpu"}
    baseline = _write(tmp_path, "baseline.json", baseline_env,
                      [_run("cpu", 1024, 512, 8, scn)])
    current = {"env": {"tex_version": "0.0", "tex_sha": "head",
                       "res": 96, "window": 48, "ticks": 4, "device": "cpu"},
               "runs": [_run("cpu", 96, 48, 4, scn)]}
    rc = b.compare(current, baseline)
    out = capsys.readouterr().out
    assert rc == 1
    assert "REFUSED" in out
    assert "res: baseline=1024" in out
    assert "window: baseline=512" in out
    assert "ticks: baseline=8" in out
    # the row diff must not have run at all -- no CHANGED/NEW ROW/GONE lines
    assert "CHANGED" not in out
    assert "NEW ROW" not in out
    assert "GONE" not in out


def test_trk100_compare_refuses_a_device_mismatch_only(tmp_path, capsys):
    b = load_counts_harness()
    scn = {"scenario_a": {"api": {"row1": _row(1)}, "frames": {}}}
    baseline_env = {"tex_version": "0.0", "tex_sha": "base",
                    "res": 96, "window": 48, "ticks": 4, "device": "cuda"}
    baseline = _write(tmp_path, "baseline.json", baseline_env,
                      [_run("cuda", 96, 48, 4, scn)])
    current = {"env": {"tex_version": "0.0", "tex_sha": "head",
                       "res": 96, "window": 48, "ticks": 4, "device": "cpu"},
               "runs": [_run("cpu", 96, 48, 4, scn)]}
    rc = b.compare(current, baseline)
    out = capsys.readouterr().out
    assert rc == 1
    assert "REFUSED" in out
    assert "device: baseline='cuda'  current='cpu'" in out
    assert "res:" not in out, "only the field(s) that actually differ are named"


def test_trk100_compare_proceeds_when_shapes_match(tmp_path, capsys):
    b = load_counts_harness()
    scn = {"scenario_a": {"api": {"row1": _row(1)}, "frames": {}}}
    env = {"tex_version": "0.0", "tex_sha": "base",
           "res": 96, "window": 48, "ticks": 4, "device": "cpu"}
    baseline = _write(tmp_path, "baseline.json", env, [_run("cpu", 96, 48, 4, scn)])
    current = {"env": dict(env, tex_sha="head"), "runs": [_run("cpu", 96, 48, 4, scn)]}
    rc = b.compare(current, baseline)
    out = capsys.readouterr().out
    assert rc == 0
    assert "REFUSED" not in out


def test_trk100_compare_does_not_refuse_an_old_baseline_missing_the_shape_fields(tmp_path, capsys):
    """Backward compatibility: a baseline saved before this landed has no
    res/window/ticks/device keys at all (`.get(f)` reads `None`). Refusing every such
    baseline would break every already-saved file the moment this lands, so a missing
    field on EITHER side is never treated as a mismatch -- only two present, disagreeing
    values are."""
    b = load_counts_harness()
    scn = {"scenario_a": {"api": {"row1": _row(1)}, "frames": {}}}
    old_env = {"tex_version": "0.0", "tex_sha": "base"}   # no shape fields at all
    baseline = _write(tmp_path, "baseline.json", old_env, [_run("cpu", 96, 48, 4, scn)])
    current = {"env": {"tex_version": "0.0", "tex_sha": "head",
                       "res": 96, "window": 48, "ticks": 4, "device": "cpu"},
               "runs": [_run("cpu", 96, 48, 4, scn)]}
    rc = b.compare(current, baseline)
    out = capsys.readouterr().out
    assert "REFUSED" not in out
    assert rc == 0
