"""
`tex validate-hw`: the determinism lane probes scatter-add programs (the path whose ordering can
vary), and the Triton lane leaves `sys.path` as it found it.
"""
import sys

from TEX_Wrangle import tex_api, tex_validate_hw as HW


def test_determinism_probes_are_scatter_adds_that_compile():
    labels = [label for label, _ in HW._DETERMINISM_PROGRAMS]
    assert labels == ["scatter", "collide"]
    for label, code in HW._DETERMINISM_PROGRAMS:
        assert "@OUT[tx, ty] +=" in code, label          # a gather cannot show atomic reordering
        assert tex_api.compile(code, HW._BINDING) is not None


def test_determinism_lane_reports_each_probe_and_the_worst(monkeypatch):
    import torch

    calls = []

    def fake_execute(prog, bindings, device="cpu", **kw):
        calls.append(prog)
        return {"OUT": torch.zeros(1, 4, 4, 3)}

    monkeypatch.setattr(tex_api, "execute", fake_execute)
    monkeypatch.setattr(HW, "_img", lambda torch_mod, side, device: torch.zeros(1, 4, 4, 3))
    res = HW._lane_determinism(torch, "cuda")
    assert res["status"] == "ran" and res["deterministic"] is True
    assert res["worst_run_to_run"] == 0.0
    assert {"worst_scatter", "worst_collide"} <= set(res)
    assert len(calls) == 2 * 5                            # a reference and four repeats per probe


def test_triton_lane_does_not_leave_the_benchmarks_dir_on_sys_path():
    before = list(sys.path)
    try:
        HW._lane_triton()
    except Exception:
        pass                                              # the lane's own result is not the subject
    assert sys.path == before
    assert str(HW._ROOT / "benchmarks") not in sys.path
