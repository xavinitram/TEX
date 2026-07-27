"""PM-7 — how fast does an interactive request get the worker, and what does the queue cost?

The v0.31 exit asks: *an interactive request preempts a running Tier-B cook and reaches first
kernel within a measured bound (target ≤ 1 stage/tile boundary, single-digit ms at proxy res),
with the abandoned Tier-B work either resumed or safely re-queued afterwards.*

Four rows, which together are the claim:

  solo                 the interactive cook alone, no queue at all. The control.
  queued_idle          the same cook through an IDLE CookQueue. The difference is the queue's
                       own tax on a request that never had to wait for anything.
  preempt_to_start     with a long SPECULATIVE cook running and a backlog behind it: from
                       `submit(INTERACTIVE)` to that cook's first tier event, and to its first
                       executed statement (the honest "first kernel" proxy — a statement that
                       reported has definitely dispatched). THIS IS PM-7.
  queued_under_load    end-to-end submit→result for the same interactive cook under that load.
                       Doc 39 §8: the queue must never tax the interactive path.

Plus the durability half: how many preempted speculative jobs were re-queued, and how many of
them finished afterwards. A preemption that loses work is not a preemption.

CUDA is synchronized inside every timed cook (the standing benchmark rule) — the queue's
worker returns as soon as Python does, so without the sync `preempt_to_start` would be
measuring kernel-launch latency and calling it a cook.

    python benchmarks/cookqueue_bench.py --res 512
    python benchmarks/cookqueue_bench.py --res 1024 --device cpu --save results/pm7.json
"""
import argparse
import json
import os
import statistics
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from TEX_Wrangle import tex_engine, tex_cookqueue as Q

# The interactive program: one cheap grade, the shape of a slider frame.
QUICK = "@OUT = vec4(@A.rgb * 1.15 + vec3(0.02), 1.0);"

# The speculative program: deliberately long, and long in STATEMENTS rather than in one big
# kernel — a preempt lands at a yield point, and yield points are per top-level statement, so
# a 200-statement program is what a fused 200-stage background render looks like to the queue.
SLOW = "\n".join(f"float v{i} = luma(@A) * {1.0 + i * 0.003};" for i in range(200)) + \
       "\n@OUT = vec4(vec3(v199), 1.0);"


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _cook(code, A, device, cancel=None, on_progress=None):
    res = tex_engine.cook(code, {"A": A}, device_mode=device, precision="fp32",
                          cancel=cancel, on_progress=on_progress)
    _sync(device)
    return res


def bench(res: int, device: str, frames: int, warmup: int) -> dict:
    torch.manual_seed(31)
    A = torch.rand(1, res, res, 4, device=device)

    # ── row 1: solo (no queue) ────────────────────────────────────────────────
    for _ in range(warmup):
        _cook(QUICK, A, device)
    solo = []
    for _ in range(frames):
        t0 = time.perf_counter()
        _cook(QUICK, A, device)
        solo.append((time.perf_counter() - t0) * 1000.0)

    # ── row 2: through an idle queue ──────────────────────────────────────────
    idle = []
    with Q.CookQueue() as q:
        for _ in range(warmup):
            q.submit(lambda c: _cook(QUICK, A, device, cancel=c), klass=Q.INTERACTIVE).result(30)
        for _ in range(frames):
            t0 = time.perf_counter()
            q.submit(lambda c: _cook(QUICK, A, device, cancel=c),
                     klass=Q.INTERACTIVE).result(timeout=60)
            idle.append((time.perf_counter() - t0) * 1000.0)

    # ── rows 3+4: under speculative load ──────────────────────────────────────
    to_tier, to_stmt, end_to_end = [], [], []
    requeued_total = resumed_total = 0
    for _ in range(frames):
        cooking = threading.Event()
        with Q.CookQueue() as q:
            spec = [q.submit(lambda c: _cook(SLOW, A, device, cancel=c,
                                             on_progress=lambda p, f: cooking.set()),
                             klass=Q.SPECULATIVE, reason=Q.NEIGHBOR_FRAME)
                    for _ in range(3)]           # one running + a backlog behind it
            if not cooking.wait(60):
                raise RuntimeError("the speculative cook never started")

            marks: dict = {}
            def on_prog(phase, frac):
                if phase == "tier" and "tier" not in marks:
                    marks["tier"] = time.perf_counter()
                elif phase == "stmt" and "stmt" not in marks:
                    marks["stmt"] = time.perf_counter()

            t0 = time.perf_counter()
            job = q.submit(lambda c: _cook(QUICK, A, device, cancel=c, on_progress=on_prog),
                           klass=Q.INTERACTIVE)
            job.result(timeout=120)
            t1 = time.perf_counter()

            end_to_end.append((t1 - t0) * 1000.0)
            if "tier" in marks:
                to_tier.append((marks["tier"] - t0) * 1000.0)
            if "stmt" in marks:
                to_stmt.append((marks["stmt"] - t0) * 1000.0)

            # The durability half: let the backlog drain and count what survived.
            q.drain(timeout=300)
            st = q.snapshot()["stats"]
            requeued_total += st["requeued"]
            resumed_total += sum(1 for j in spec if j.state == Q.DONE)

    def med(xs):
        return round(statistics.median(xs), 3) if xs else None

    out = {
        "res": res, "device": device, "frames": frames,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": ("sm_" + "".join(map(str, torch.cuda.get_device_capability(0))))
                      if torch.cuda.is_available() else None,
        "solo_ms": med(solo),
        "queued_idle_ms": med(idle),
        "preempt_to_tier_ms": med(to_tier),
        "preempt_to_first_stmt_ms": med(to_stmt),
        "queued_under_load_ms": med(end_to_end),
        "speculative_requeued": requeued_total,
        "speculative_resumed": resumed_total,
        "speculative_submitted": 3 * frames,
    }
    out["queue_tax_ms"] = round(out["queued_idle_ms"] - out["solo_ms"], 3)
    out["load_penalty_x"] = round(out["queued_under_load_ms"] / max(out["solo_ms"], 1e-9), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=512, help="proxy resolution (PM-7 says 'proxy')")
    ap.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--frames", type=int, default=9)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--save", type=str, default=None)
    a = ap.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device

    r = bench(a.res, device, a.frames, a.warmup)
    print(f"\nPM-7 — cook-queue preemption  ({r['res']}^2, {r['device']}"
          f"{', ' + r['capability'] if r['capability'] else ''})")
    print(f"  solo cook (no queue)          {r['solo_ms']:8.3f} ms")
    print(f"  through an idle queue         {r['queued_idle_ms']:8.3f} ms   "
          f"(queue tax {r['queue_tax_ms']:+.3f} ms)")
    print(f"  submit -> tier start          {r['preempt_to_tier_ms']:8.3f} ms   <- PM-7")
    print(f"  submit -> first statement     {r['preempt_to_first_stmt_ms']:8.3f} ms")
    print(f"  submit -> result, under load  {r['queued_under_load_ms']:8.3f} ms   "
          f"({r['load_penalty_x']}x solo)")
    print(f"  speculative: {r['speculative_submitted']} submitted, "
          f"{r['speculative_requeued']} re-queued, {r['speculative_resumed']} resumed to DONE")
    if a.save:
        path = a.save if os.path.isabs(a.save) else \
            os.path.join(os.path.dirname(os.path.abspath(__file__)), a.save)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2)
        print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
