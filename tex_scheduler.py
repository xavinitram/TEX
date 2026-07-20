"""
SCHED-2 — device-placement scheduler.

Where should each node of a TEX graph cook — CPU or CUDA? A node's cook is faster on the GPU
but moving its input/output across PCIe is not free, so the answer is a trade: minimize
`Σ cook_cost(node, device) + Σ transfer_cost(edge)` subject to user device pins and per-device
memory budgets. This module is that planner. It does NOT cook, does NOT import `comfy`
(PORT-1, invariant #8), and never touches a `torch.cuda.*` API without a capability check —
device/memory facts come through `tex_runtime.host`.

There is no multi-node executor yet (GRAPH-1), so SCHED-2 ships MEASURED, TESTED, and DORMANT
(the ROI-3 / CACHE-2 posture): its consumer is a GraphSpec-emitting host (PORT-5 / GRAPH-1)
that feeds the plan's per-node device into each node's `tex_engine.prepare(device_mode=...)`.
The default single-node ComfyUI path never reaches it (invariant #7).

Design (mirrors autotier's testable-state discipline — costs are explicit numbers a test can
feed):
  * The solver is exact where it can be and greedy where it must be. A LINEAR chain gets a
    Viterbi DP (exact, O(N·D²)); a small DAG gets exact enumeration; a large DAG falls back to
    GREEDY. Greedy — "a node runs where its input already is, else the default device" — is
    `resolve_device`'s own auto rule (tex_engine.resolve_device), so it is the CORRECTNESS
    BASELINE, not a guess: when cost data is missing (cold autotier) or a node is pinned, the
    plan defers to it rather than invent a placement from a zero cost (the trap that would pile
    everything onto one unmeasured device).
  * Device is part of every result key (CACHE-1: `lineage_key` folds it), so a placement
    decision is VISIBLE. A plan is therefore FROZEN per render range and re-solved only at an
    interactive/idle boundary — moving a node between devices mid-sequence pops pixels by the
    invariant-#9 envelope (up to 6.1e-2) and is a REJECTED decision (roadmap §7). Hysteresis
    (`hysteresis_ms`) damps interactive flapping between two near-equal plans.
  * Distinct from the rejected PF-4: this chooses WHERE a node runs, never WHICH TIER.
  * Scope: CUDA + CPU. The ENG-8 transfer model has one CUDA device (no D2D lane), so a
    `cuda:i → cuda:j` edge is out of its competence and treated as pinned/greedy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


# ── The graph the scheduler consumes ─────────────────────────────────────────
# A node is one TEX program (a graph node / fused region). The host builds these — the
# scheduler stays pure so a test feeds explicit costs, exactly as autotier takes explicit ms.

@dataclass(frozen=True)
class SchedNode:
    """One placeable node. `inputs` are the ids of its graph predecessors (edges). Cost inputs
    are best-effort: a `None`/0 means 'unknown' and the planner degrades to greedy for it
    rather than treating unknown as free."""
    id: Any
    program_fp: str | None = None       # value-independent fp → autotier cook-cost key (None → proxy)
    spatial_shape: tuple | None = None  # (B, H, W): the autotier resolution bucket + byte sizing
    precision: str = "fp32"
    out_nbytes: int = 0                 # this node's output size — the per-edge transfer payload
    peak_bytes: int = 0                 # est. cook peak (memory-budget fit); 0 = unknown → always fits
    inputs: tuple = ()                  # predecessor node ids
    pin: str | None = None              # user device pin ("cpu" | "cuda" | "cuda:N"), or None


@dataclass(frozen=True)
class Placement:
    """A frozen plan: `devices` maps every node id → its chosen device string. `est_cost_ms` is
    the planner's own cost estimate (cook + transfer); `method` records how it was solved
    (`dp` | `enumerate` | `greedy`), and `note` any fallback reason — so a host can log WHY a
    placement is what it is (the CACHE-1 visibility contract)."""
    devices: dict
    est_cost_ms: float = 0.0
    method: str = "greedy"
    note: str = ""


# ── Device set + default cost providers ──────────────────────────────────────

def available_devices() -> list[str]:
    """The placement targets: always CPU, plus CUDA when the box has it. Uses the bare "cuda"
    label (the ENG-8 model is single-CUDA-device; a multi-GPU host pins per index instead).
    Capability-checked (invariant / roadmap §7: no bare torch.cuda import path)."""
    devs = ["cpu"]
    try:
        import torch
        if torch.cuda.is_available():
            devs.append("cuda")
    except Exception:
        pass
    return devs


def _dev_type(device: str) -> str:
    return "cuda" if str(device).startswith("cuda") else "cpu"


def default_cook_cost(node: SchedNode, device: str) -> float | None:
    """cook_cost(node, device) from autotier's persisted medians (SCHED-2's `cook_cost`). None
    when this program was never measured on that (device-type, precision, resolution) — the
    caller then falls to the greedy baseline for that node rather than inventing a number."""
    if node.program_fp is None or node.spatial_shape is None:
        return None
    try:
        from .tex_runtime import autotier
        key = autotier.make_key(node.program_fp, _dev_type(device), node.precision,
                                node.spatial_shape)
        return autotier.cook_ms(key)
    except Exception:
        return None


def default_transfer_cost(nbytes: int, src_dev: str, dst_dev: str) -> float:
    """transfer_cost(edge) from the ENG-8 probe. Zero within a device class (CPU→CPU, or
    same-index CUDA→CUDA — no D2D lane in the model); H2D for CPU→CUDA, D2H for CUDA→CPU. The
    ≥1 MiB egress-pin threshold (docs/xpu-transfer-scheduling.md) decides the pinned lane."""
    s, d = _dev_type(src_dev), _dev_type(dst_dev)
    if s == d:
        return 0.0
    if not nbytes:
        return 0.0
    try:
        from .tex_runtime.xfer import transfer_ms
        direction = "h2d" if s == "cpu" else "d2h"
        pinned = nbytes >= (1 << 20)
        return transfer_ms(int(nbytes), pinned, direction)
    except Exception:
        return 0.0


# ── Candidate device sets (pins + memory budget) ─────────────────────────────

def _free_bytes(device: str) -> float | None:
    try:
        import torch
        from .tex_runtime.host import get_host_services
        return get_host_services().get_free_memory(torch.device(device))
    except Exception:
        return None


def _candidates(node: SchedNode, devices: list[str]) -> list[str]:
    """The devices a node MAY run on: its pin alone if pinned (a pin is honoured even if it
    over-budgets — the user asked), else every available device the node's estimated peak fits
    in. A CUDA device that can't be sized (no free-mem answer) is kept — refusing a placement
    for lack of data is the wrong direction; an actual OOM is the engine's ladder to catch."""
    if node.pin is not None:
        return [node.pin]
    out = []
    for dev in devices:
        if node.peak_bytes and _dev_type(dev) == "cuda":
            free = _free_bytes(dev)
            if free is not None and node.peak_bytes > free:
                continue        # provably won't fit — drop this device for this node
        out.append(dev)
    return out or ["cpu"]       # never leave a node with nowhere to run


# ── Topology ─────────────────────────────────────────────────────────────────

def _toposort(nodes: list[SchedNode]) -> list[SchedNode] | None:
    """Kahn topological order, or None if the graph has a cycle or a dangling input (either is
    a malformed graph the planner refuses — the host falls back to per-node greedy)."""
    by_id = {n.id: n for n in nodes}
    if len(by_id) != len(nodes):
        return None                      # duplicate ids
    indeg = {n.id: 0 for n in nodes}
    succ: dict = {n.id: [] for n in nodes}
    for n in nodes:
        for u in n.inputs:
            if u not in by_id:
                return None              # dangling edge
            indeg[n.id] += 1
            succ[u].append(n.id)
    ready = [nid for nid, d in indeg.items() if d == 0]
    order = []
    while ready:
        nid = ready.pop()
        order.append(by_id[nid])
        for v in succ[nid]:
            indeg[v] -= 1
            if indeg[v] == 0:
                ready.append(v)
    return order if len(order) == len(nodes) else None


def _is_linear_chain(order: list[SchedNode]) -> bool:
    """True only for a SINGLE connected simple chain: exactly one source (a node with no inputs),
    every node ≤1 input and ≤1 consumer — the shape a linear fusion producer emits, where the
    Viterbi DP is exact and cheap. A FOREST (≥2 disconnected chains — a dead/unconsumed stage in a
    GraphSpec produces one) has ≥2 sources and is NOT linear: the Viterbi reconstruction assumes a
    single source, so a second source's None back-pointer would crash it. A forest routes to exact
    enumeration or greedy instead (both handle disconnected graphs)."""
    return (sum(1 for n in order if not n.inputs) == 1
            and not any(len(n.inputs) > 1 for n in order)
            and all(c <= 1 for c in _consumer_count(order).values()))


# ── Cost of a full assignment ────────────────────────────────────────────────

def _cook(node, device, cook_cost, greedy_dev) -> float:
    c = cook_cost(node, device)
    if c is not None:
        return c
    # Unmeasured: a proxy that still prefers the greedy device (a small penalty for deviating),
    # so the solver never treats 'unknown' as 0 and piles onto an unmeasured device.
    base = max(1e-3, node.out_nbytes / 1e9)     # ~GB/ms scale placeholder, strictly positive
    return base if device == greedy_dev[node.id] else base * 1.5   # greedy_dev is fully populated


def _consumer_count(order) -> dict:
    c: dict = {n.id: 0 for n in order}
    for n in order:
        for u in n.inputs:
            c[u] = c.get(u, 0) + 1
    return c


def _assignment_cost(order, dev_of, by_id, cook_cost, transfer_cost, greedy_dev,
                     boundary: str | None = None) -> float:
    """Total wall-cost of an assignment: every node's cook + every graph edge's transfer +
    the BOUNDARY transfers — a source's external input arrives on `boundary` (an H2D if the
    source runs on CUDA) and a sink's output must return to `boundary` (a D2H). Without the
    boundary terms the solver would trivially pile everything onto the fastest cook device,
    ignoring the up/download the greedy baseline avoids by staying put. `out_nbytes` sizes the
    boundary payloads (input≈output for the image ops this targets)."""
    total = 0.0
    consumers = _consumer_count(order) if boundary is not None else None
    for n in order:
        total += _cook(n, dev_of[n.id], cook_cost, greedy_dev)
        for u in n.inputs:
            total += transfer_cost(by_id[u].out_nbytes, dev_of[u], dev_of[n.id])
        if boundary is not None:
            if not n.inputs:                                   # source: external input uploaded
                total += transfer_cost(n.out_nbytes, boundary, dev_of[n.id])
            if consumers.get(n.id, 0) == 0:                    # sink: result returned to host
                total += transfer_cost(n.out_nbytes, dev_of[n.id], boundary)
    return total


# ── Greedy baseline (the correctness fallback = resolve_device's auto rule) ───

def _greedy(order, by_id, cand, default_device) -> dict:
    """Each node runs where its input already is, else the default device — pins win. This mirrors
    `tex_engine.resolve_device`'s auto rule faithfully: land on GPU if ANY input is already on GPU
    (a fan-in node follows the accelerator, not merely its first edge — the resolve_device
    contract), else follow the first input's device class. The chosen class is matched to a
    candidate by DEVICE TYPE, so a `cuda:0` source and a bare `cuda` candidate are the same class
    (an exact-string membership would spuriously drop a GPU-pinned node's consumer to CPU with a
    needless D2H). This is the correctness baseline every other path defers to."""
    dev: dict = {}

    def _match(src_devs, node):
        # `_dev_type` is binary, so "no input on cuda" ⇒ every input is cpu-class ⇒ want cpu.
        want = "cuda" if any(_dev_type(d) == "cuda" for d in src_devs) else "cpu"
        return next((c for c in cand[node.id] if _dev_type(c) == want), cand[node.id][0])

    for n in order:
        if n.pin is not None:
            dev[n.id] = n.pin
        elif n.inputs:
            dev[n.id] = _match([dev.get(u, default_device) for u in n.inputs], n)
        else:
            dev[n.id] = _match([default_device], n)
    return dev


# ── Exact solvers ────────────────────────────────────────────────────────────

def _viterbi_chain(order, by_id, cand, cook_cost, transfer_cost, greedy_dev, boundary) -> dict:
    """Exact min-cost placement for a LINEAR chain: a Viterbi pass over the ≤|devices| states
    per node. `best[dev]` = min cost of the prefix ending with this node on `dev`; back-pointers
    reconstruct the assignment. Boundary transfers (source upload, sink download to `boundary`)
    fold into the source's base and the sink's per-device cost so the DP stays exact vs
    `_assignment_cost`."""
    last = len(order) - 1
    prev_cost: dict = {}
    prev_ptr: list = []
    for i, n in enumerate(order):
        cur_cost: dict = {}
        ptr: dict = {}
        pred = n.inputs[0] if n.inputs else None
        for d in cand[n.id]:
            ck = _cook(n, d, cook_cost, greedy_dev)
            if boundary is not None and i == last:            # sink: return the result to host
                ck += transfer_cost(n.out_nbytes, d, boundary)
            if pred is None:
                if boundary is not None:                      # source: upload the external input
                    ck += transfer_cost(n.out_nbytes, boundary, d)
                cur_cost[d] = ck
                ptr[d] = None
            else:
                best_prev, best_pd = None, None
                for pd, pc in prev_cost.items():
                    edge = transfer_cost(by_id[pred].out_nbytes, pd, d)
                    cand_cost = pc + edge
                    if best_prev is None or cand_cost < best_prev:
                        best_prev, best_pd = cand_cost, pd
                cur_cost[d] = ck + (best_prev or 0.0)
                ptr[d] = best_pd
        prev_cost = cur_cost
        prev_ptr.append(ptr)
    # reconstruct
    dev: dict = {}
    d = min(prev_cost, key=prev_cost.get)
    for i in range(len(order) - 1, -1, -1):
        dev[order[i].id] = d
        d = prev_ptr[i][d]   # None only at the source (i==0); a real device for every i>0
    return dev


_ENUM_CAP = 50000   # exact enumeration only while Π|candidates| stays cheap; else greedy


def _enumerate(order, by_id, cand, cook_cost, transfer_cost, greedy_dev, boundary) -> dict | None:
    """Exact min-cost placement for a small (possibly branching) DAG by enumerating every
    candidate combination. Returns None when the product of candidate-set sizes exceeds
    `_ENUM_CAP` (→ greedy). Fusion regions are small, so the realistic DAG is solved exactly."""
    total = 1
    for n in order:
        total *= len(cand[n.id])
        if total > _ENUM_CAP:
            return None
    ids = [n.id for n in order]
    choices = [cand[nid] for nid in ids]
    best_dev, best_cost = None, None
    idx = [0] * len(ids)
    while True:
        dev_of = {ids[k]: choices[k][idx[k]] for k in range(len(ids))}
        cost = _assignment_cost(order, dev_of, by_id, cook_cost, transfer_cost, greedy_dev, boundary)
        if best_cost is None or cost < best_cost:
            best_cost, best_dev = cost, dev_of
        # odometer increment
        j = len(ids) - 1
        while j >= 0:
            idx[j] += 1
            if idx[j] < len(choices[j]):
                break
            idx[j] = 0
            j -= 1
        if j < 0:
            break
    return best_dev


# ── The planner ──────────────────────────────────────────────────────────────

def plan_placement(nodes: list[SchedNode], *, devices: list[str] | None = None,
                   default_device: str = "cpu",
                   cook_cost: Callable = default_cook_cost,
                   transfer_cost: Callable = default_transfer_cost,
                   previous: Placement | None = None,
                   hysteresis_ms: float = 0.0) -> Placement:
    """Solve device placement for `nodes`. Returns a `Placement`.

    `devices` defaults to `available_devices()`. `default_device` is where a source node's
    external input lives (the greedy anchor). `cook_cost`/`transfer_cost` are injectable so a
    test drives the solver with explicit numbers (autotier/xfer wired by default). `previous` +
    `hysteresis_ms` apply frozen-per-range damping: a node keeps its previous device unless the
    fresh plan beats keeping it by more than `hysteresis_ms` total (anti-flapping).

    Never raises: any structural surprise (cycle, dangling edge, empty graph) falls to the
    greedy baseline with a `note`."""
    devices = devices or available_devices()
    if not nodes:
        return Placement({}, 0.0, "greedy", "empty graph")
    order = _toposort(nodes)
    by_id = {n.id: n for n in nodes}
    if order is None:
        # Malformed graph: place each node on its pin or the default, honestly noted.
        dev = {n.id: (n.pin or (default_device if default_device in devices else "cpu"))
               for n in nodes}
        return Placement(dev, 0.0, "greedy", "unorderable graph (cycle/dangling) -> per-node default")

    cand = {n.id: _candidates(n, devices) for n in order}
    greedy_dev = _greedy(order, by_id, cand, default_device)

    method, note = "greedy", ""
    if len(devices) < 2:
        dev, method, note = greedy_dev, "greedy", "single device"
    elif _is_linear_chain(order):
        dev, method = _viterbi_chain(order, by_id, cand, cook_cost, transfer_cost,
                                     greedy_dev, default_device), "dp"
    else:
        enum = _enumerate(order, by_id, cand, cook_cost, transfer_cost, greedy_dev, default_device)
        if enum is not None:
            dev, method = enum, "enumerate"
        else:
            dev, method, note = greedy_dev, "greedy", "graph too large for exact solve"

    if previous is not None and hysteresis_ms > 0:
        dev = _apply_hysteresis(order, by_id, dev, previous.devices, cand,
                                cook_cost, transfer_cost, greedy_dev, hysteresis_ms, default_device)
        note = (note + "; " if note else "") + "hysteresis"

    cost = _assignment_cost(order, dev, by_id, cook_cost, transfer_cost, greedy_dev, default_device)
    return Placement(dev, cost, method, note)


def _apply_hysteresis(order, by_id, fresh, prev, cand, cook_cost, transfer_cost,
                      greedy_dev, hysteresis_ms, boundary) -> dict:
    """Keep a node on its PREVIOUS device unless switching to the fresh plan's device improves
    total cost by more than `hysteresis_ms` — damps interactive flapping between near-equal
    plans (the roadmap's anti-flap requirement). A node with no previous placement, or whose
    previous device is no longer a candidate (pin/budget change), takes the fresh choice."""
    kept = dict(fresh)
    cost_fresh = _assignment_cost(order, kept, by_id, cook_cost, transfer_cost, greedy_dev, boundary)
    for n in order:
        p = prev.get(n.id)
        if p is None or p not in cand[n.id] or p == kept.get(n.id):
            continue
        trial = {**kept, n.id: p}
        cost_keep = _assignment_cost(order, trial, by_id, cook_cost, transfer_cost, greedy_dev, boundary)
        if cost_keep - cost_fresh <= hysteresis_ms:
            kept, cost_fresh = trial, cost_keep     # within the dead-band → keep the old device
    return kept


class Scheduler:
    """A stateful convenience wrapper enforcing frozen-per-range planning: `plan()` re-solves
    and remembers the result; within a render range the host reuses the returned `Placement`
    and only calls `plan()` again at an interactive/idle boundary (never mid-sequence — the
    §7 rejection). `hysteresis_ms` damps flapping across those re-plans."""
    def __init__(self, devices: list[str] | None = None, *, hysteresis_ms: float = 0.0):
        self.devices = devices or available_devices()
        self.hysteresis_ms = hysteresis_ms
        self.current: Placement | None = None

    def plan(self, nodes: list[SchedNode], *, default_device: str = "cpu", **kw) -> Placement:
        self.current = plan_placement(
            nodes, devices=self.devices, default_device=default_device,
            previous=self.current, hysteresis_ms=self.hysteresis_ms, **kw)
        return self.current

    def reset(self) -> None:
        """Drop the frozen plan — the next `plan()` solves fresh (a new project / render range)."""
        self.current = None


# ── GraphSpec adapter (structure + the new per-stage device pin) ──────────────

def graph_from_spec(spec: dict, *, spatial_shape: tuple | None = None,
                    precision: str = "fp32") -> list[SchedNode]:
    """Build the scheduler's node list from a SCHED-1 GraphSpec (`_tex_chain`), reading its
    structure (stages + terminal, linear or DAG edges) and the ADDITIVE per-stage/terminal
    `device` pin SCHED-2 introduces (an optional key older readers ignore — no schema bump).
    Cost fields (`program_fp`, `out_nbytes`, `peak_bytes`) are left for the host to fill once
    it has compiled/sized each stage — this adapter supplies the graph shape and the pins.
    Node ids are `0..N-1` for stages and `"terminal"` for the sink. Never raises → [] on a
    malformed spec."""
    try:
        stages = spec.get("stages") or []
        is_dag = bool(spec.get("dag"))
        nodes: list[SchedNode] = []
        for i, st in enumerate(stages):
            inputs: tuple = ()
            if is_dag:
                ci = st.get("chain_inputs") or {}
                inputs = tuple(sorted({src for (src, _out) in ci.values()}))
            elif i > 0:
                inputs = (i - 1,)     # linear: source-first chain
            nodes.append(SchedNode(id=i, spatial_shape=spatial_shape, precision=precision,
                                   inputs=inputs, pin=st.get("device")))
        # terminal consumes the last stage (linear) or its terminal_chain_inputs (DAG)
        if is_dag:
            tci = spec.get("terminal_chain_inputs") or {}
            t_inputs = tuple(sorted({src for (src, _out) in tci.values()}))
        else:
            t_inputs = (len(stages) - 1,) if stages else ()
        nodes.append(SchedNode(id="terminal", spatial_shape=spatial_shape, precision=precision,
                               inputs=t_inputs, pin=spec.get("terminal_device")))
        return nodes
    except Exception:
        return []
