"""TRK-133 — `benchmarks/roi_codegen_ab_bench.py` exists and its pure helpers behave.

The script itself is a measurement tool (it drives real cooks, on CPU or CUDA, over
tens of rounds) — this file does not re-run that sitting. It pins the parts that are
cheap to check and load-bearing for the "commit the method, not the machine" goal
`docs/brief-conventions.md` names: the box name is derived at runtime rather than
hard-coded, the cache root defaults under the system temp dir rather than a private
path, and the null-control math (the same split-half comparison the maintainer's
measurement script used) is correct.

Loaded by path, like `helpers.load_counts_harness` loads `host_path_counts.py`:
`benchmarks/` is `.comfyignore`d and not a package, so there is no import name.
"""
import importlib.util
import os
import sys
import tempfile

import pytest


def _load_roi_codegen_ab_bench():
    mod = sys.modules.get("_trk133_roi_codegen_ab_bench")
    if mod is not None:
        return mod
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(pkg_dir, "benchmarks", "roi_codegen_ab_bench.py")
    spec = importlib.util.spec_from_file_location("_trk133_roi_codegen_ab_bench", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_trk133_roi_codegen_ab_bench"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def b():
    return _load_roi_codegen_ab_bench()


def test_trk133_module_loads_and_defines_the_v030_shapes(b):
    assert b.SHAPES == [(256, 1024), (1024, 2048)], \
        "these are the shapes docs/roi-spatial-laziness.md's figure names"


def test_trk133_box_name_is_derived_not_hard_coded(b):
    """The whole point of committing this script: it names ITS OWN box at runtime,
    never a literal machine name. Verified two ways: no box-shaped literal string sits
    in the source, and the function's return value tracks what torch itself reports."""
    import inspect
    src = inspect.getsource(b._box_name)
    assert "RTX" not in src and "Blackwell" not in src and "Turing" not in src, \
        "a literal box name in the source would defeat this script's own purpose"
    name = b._box_name()
    assert isinstance(name, str) and name
    import torch
    if torch.cuda.is_available():
        assert torch.cuda.get_device_name(0) in name
    else:
        assert "CPU" in name


def test_trk133_cache_root_defaults_under_the_system_temp_dir(b):
    """No private path is hard-coded: the default cache root is derived from
    `tempfile.gettempdir()`, and an explicit override is honoured verbatim."""
    default = b._cache_root(None)
    assert default.startswith(tempfile.gettempdir())
    explicit = b._cache_root(os.path.join(tempfile.gettempdir(), "somewhere_else"))
    assert explicit == os.path.join(tempfile.gettempdir(), "somewhere_else")


def test_trk133_null_spread_is_the_split_half_self_comparison(b):
    xs = [1.0, 1.0, 3.0, 3.0]           # halves: [1.0, 1.0] and [3.0, 3.0]
    ma, mb, ratio = b._null_spread(xs)
    assert ma == 1.0 and mb == 3.0
    assert abs(ratio - (1.0 / 3.0)) < 1e-9


def test_trk133_null_spread_of_identical_halves_is_one(b):
    xs = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    ma, mb, ratio = b._null_spread(xs)
    assert abs(ratio - 1.0) < 1e-9


def test_trk133_no_machine_or_host_path_literal_in_the_source(b):
    """A second, source-level guard alongside `test_simp3_no_machine_paths.py`: this
    script's whole reason for existing is that a prior lane's version of it hard-coded
    a worktree path, a cache directory and a box name that only existed on ONE machine
    (see the module docstring). Fail loudly here too if that regresses."""
    import inspect
    from pathlib import Path
    src = inspect.getsource(b)
    assert "TEX_wt" not in src
    # Derived, not hard-coded (the same principle this whole test file checks the SCRIPT
    # for): whatever this box's own username is must not appear in the script's source,
    # but the username itself is never spelled out in this pushed file either.
    username = Path.home().name.lower()
    if username:
        assert username not in src.lower()
    assert "comfyui_windows_portable" not in src.lower()
