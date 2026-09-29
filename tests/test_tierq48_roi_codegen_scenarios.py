"""TIERQ-48 — the realistic-viewport scenarios added to `benchmarks/roi_codegen_ab_bench.py`.

The re-measurement this ask needs is at a REALISTIC interactive-viewport shape (a small,
16:9 window against a 1920x1080 or 3840x2160/"4k" canvas) rather than the square crop of a
square canvas TRK-133's own `SHAPES` table used. `run_shape` (the TRK-133-pinned entry
point) is now a thin wrapper over the new `run_shape_rect`, and `INTERACTIVE_SCENARIOS` is
an ADDITIVE second table — this file pins that the addition is genuinely additive: the
old table/call shape is untouched, and the new one produces the documented areas.

Loaded by path, like `test_trk133_roi_codegen_ab_bench.py` loads the same module —
`benchmarks/` is `.comfyignore`d and not a package, so there is no import name.
"""
import pytest

from helpers import load_benchmark


def _load():
    return load_benchmark("roi_codegen_ab_bench.py")


@pytest.fixture(scope="module")
def b():
    return _load()


def test_tierq48_shapes_table_is_untouched(b):
    """The TRK-133 pin, re-checked here: adding the new table must not have moved the
    old one."""
    assert b.SHAPES == [(256, 1024), (1024, 2048)]


def test_tierq48_interactive_scenarios_are_roughly_six_percent_16x9(b):
    for name, (roi_w, roi_h, canvas_w, canvas_h) in b.INTERACTIVE_SCENARIOS.items():
        canvas_area = canvas_w * canvas_h
        roi_area = roi_w * roi_h
        frac = roi_area / canvas_area
        assert 0.055 <= frac <= 0.065, f"{name}: roi is {frac:.4f} of the canvas, want ~0.06"
        # same aspect ratio as its own canvas (16:9), within rounding
        assert abs((roi_w / roi_h) - (canvas_w / canvas_h)) < 0.05, \
            f"{name}: roi aspect does not match its canvas aspect"


def test_tierq48_interactive1080p_and_4k_are_the_named_canvases(b):
    assert b.INTERACTIVE_SCENARIOS["interactive1080p"][2:4] == (1920, 1080)
    assert b.INTERACTIVE_SCENARIOS["4k"][2:4] == (3840, 2160)


def test_tierq48_run_shape_square_wrapper_matches_run_shape_rect(b, monkeypatch, tmp_path):
    """`run_shape(roi_side, res, ...)` must still behave exactly like calling
    `run_shape_rect(roi_side, roi_side, res, res, ...)` — the refactor is a pure
    delegation, not a re-implementation that could silently drift."""
    calls = []
    real = b.run_shape_rect

    def _spy(roi_w, roi_h, canvas_w, canvas_h, *a, **kw):
        calls.append((roi_w, roi_h, canvas_w, canvas_h))
        return real(roi_w, roi_h, canvas_w, canvas_h, *a, **kw)

    monkeypatch.setattr(b, "run_shape_rect", _spy)
    b.run_shape(64, 256, "cpu", str(tmp_path), "tierq48test", 3)
    assert calls == [(64, 64, 256, 256)]


def test_tierq48_scenario_cli_flag_exists_and_defaults_to_square(b, monkeypatch):
    import argparse
    captured = {}

    def _capture(self, args=None, namespace=None):
        captured["parser"] = self
        raise SystemExit(0)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", _capture)
    with pytest.raises(SystemExit):
        b.main([])
    parser = captured["parser"]
    assert parser.get_default("scenario") == "square"
    choices = next(a.choices for a in parser._actions if a.dest == "scenario")
    assert tuple(choices) == ("square", "interactive1080p", "4k", "all")
