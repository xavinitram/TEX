"""v0.52 sweep: scheduler rows (mixed measured/unmeasured costs, hysteresis bound,
per-plan free-memory snapshot, memoised providers)."""
from TEX_Wrangle import tex_scheduler as S
from TEX_Wrangle.tex_scheduler import SchedNode, plan_placement

MB = 1 << 20
DEVS = ["cpu", "cuda"]


def _chain(n):
    return [SchedNode(id=i, out_nbytes=MB, inputs=((i - 1,) if i > 0 else ())) for i in range(n)]


def _xfer(c):
    return lambda nb, s, d: (0.0 if str(s).startswith("cuda") == str(d).startswith("cuda") else c)


def test_half_measured_node_stays_on_the_greedy_device():
    # cuda is measured at 3 ms; cpu was never measured. The old placeholder (~0.001 ms)
    # made cpu look free and pulled the node there.
    def cook(node, dev):
        return 3.0 if str(dev).startswith("cuda") else None

    nodes = _chain(1)
    p = plan_placement(nodes, devices=DEVS, default_device="cuda",
                       cook_cost=cook, transfer_cost=_xfer(0.0))
    assert p.devices == {0: "cuda"}
    p = plan_placement(_chain(3), devices=DEVS, default_device="cuda",
                       cook_cost=cook, transfer_cost=_xfer(0.5))
    assert set(p.devices.values()) == {"cuda"}


def test_fully_measured_nodes_still_move_to_the_cheaper_device():
    p = plan_placement(_chain(3), devices=DEVS, default_device="cpu",
                       cook_cost=lambda n, d: 1.0 if str(d).startswith("cuda") else 10.0,
                       transfer_cost=_xfer(1.0))
    assert set(p.devices.values()) == {"cuda"}


def test_hysteresis_bounds_the_total_regression():
    # Keeping the old cpu device costs 1 ms more per node; with a 2 ms dead-band only the
    # first two keeps fit. The old code moved the baseline after every keep and kept all six.
    n = 6
    prev = S.Placement({i: "cpu" for i in range(n)}, 0.0, "dp", "")

    def cook(node, dev):
        return 1.0 if str(dev).startswith("cuda") else 2.0

    nodes = [SchedNode(id=i, out_nbytes=MB) for i in range(n)]      # independent nodes
    fresh = plan_placement(nodes, devices=DEVS, default_device="cpu",
                           cook_cost=cook, transfer_cost=_xfer(0.0))
    assert set(fresh.devices.values()) == {"cuda"}
    p = plan_placement(nodes, devices=DEVS, default_device="cpu", cook_cost=cook,
                       transfer_cost=_xfer(0.0), previous=prev, hysteresis_ms=2.0)
    assert p.est_cost_ms - fresh.est_cost_ms <= 2.0 + 1e-9
    assert sum(1 for d in p.devices.values() if d == "cpu") == 2


def test_free_memory_is_asked_once_per_device_per_plan(monkeypatch):
    calls = []

    def free(dev):
        calls.append(dev)
        return 10 ** 12

    monkeypatch.setattr(S, "_free_bytes", free)
    nodes = [SchedNode(id=i, out_nbytes=MB, peak_bytes=MB, inputs=((i - 1,) if i else ()))
             for i in range(5)]
    plan_placement(nodes, devices=DEVS, default_device="cpu",
                   cook_cost=lambda n, d: 1.0, transfer_cost=_xfer(1.0))
    assert calls == ["cuda"]


def test_enumeration_asks_each_provider_once_per_key():
    cook_calls, xfer_calls = [], []

    def cook(node, dev):
        cook_calls.append((node.id, dev))
        return 1.0

    def xfer(nb, s, d):
        xfer_calls.append((nb, s, d))
        return 0.0 if s == d else 1.0

    diamond = [SchedNode(id=0, out_nbytes=MB), SchedNode(id=1, out_nbytes=MB, inputs=(0,)),
               SchedNode(id=2, out_nbytes=MB, inputs=(0,)),
               SchedNode(id=3, out_nbytes=MB, inputs=(1, 2))]
    p = plan_placement(diamond, devices=DEVS, default_device="cpu",
                       cook_cost=cook, transfer_cost=xfer)
    assert p.method == "enumerate"
    assert len(cook_calls) == len(set(cook_calls))
    assert len(xfer_calls) == len(set(xfer_calls))
