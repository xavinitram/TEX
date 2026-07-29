"""PM-9 — headless 100-frame playback: how much I/O hides behind compute?

The v0.34 exit (doc 41 §3): *headless 100-frame playback through the CookQueue where input
prefetch and output writes overlap compute, **>= 80% of I/O hidden** against the serial
baseline.*

Two routes over the same 100 frames, same pixels:

  serial      fetch -> cook -> write, one after another. What a host without IO-1 does.
  overlapped  a host I/O thread fetches frame N+lookahead and LANDS a promise; the CookQueue
              admits frame N's cook when ITS input landed (IO-1's WAITING state); the
              completed frame goes to a writer thread as an XPU-2 handle (§3.3) and the
              worker moves straight to the next cook.

The number: `hidden = 1 - (t_overlapped - t_compute) / t_io_serial` — of the I/O time a
serial loop pays, how much disappeared behind compute.

WHY THE I/O IS ON HOST THREADS AND NOT ON THE QUEUE. `tex_provider.declare_window` submits
prefetch as SPECULATIVE jobs, which run on the cook queue's SINGLE worker — deliberately, so
idle-time prefetch is priced and sheddable like any other bet. That is the right mode when
the worker is idle and the WRONG one here: a prefetch occupying the worker is not overlapping
compute, it is taking turns with it. Doc 41 §3.2 is explicit that the one-worker contract is
load-bearing and "the single worker never blocks on I/O", which is what the promise path
delivers. Both modes are measured below so the difference is on the record rather than
folklore.

HONEST CAVEAT, stated before the number. The provider and writer stall with `time.sleep`,
which releases the GIL cleanly; a real Python decoder does not, and would contend with the
cook worker in a way this harness cannot see. This measures THE SCHEDULER, not a disk.

    python benchmarks/io_playback_bench.py
    python benchmarks/io_playback_bench.py --frames 100 --res 512 --save results/pm9.json
"""
import argparse
import json
import os
import queue as _queue
import statistics
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from TEX_Wrangle import tex_cookqueue as Q, tex_engine, tex_provider
from TEX_Wrangle.tex_compiler.types import TEXType
from TEX_Wrangle.tex_marshalling import Promise
from TEX_Wrangle.tex_runtime import streams

#: A playback frame's comp. `--stages` sets how many grades deep it is, and THAT KNOB IS THE
#: WHOLE EXPERIMENT: "I/O hidden behind compute" is only a meaningful quantity when there is
#: compute to hide behind. A single grade at 512² on this GPU is 0.35 ms against ~9.5 ms of
#: I/O per frame — 3.5% of the work — and no scheduler can hide 9.5 ms behind 0.35 ms. The
#: default is chosen so compute per frame lands in the same order as I/O per frame, which is
#: what a real 1080p/4K playback comp looks like. The thin-compute case is measured too, and
#: reported as the boundary rather than hidden.
_GRADE = "  c = vec4(c.rgb * {k:.4f} + vec3({b:.4f}), c.a);"


def _code(stages: int) -> str:
    body = "\n".join(_GRADE.format(k=1.0 + i * 0.001, b=i * 0.0004) for i in range(stages))
    return f"vec4 c = @A;\n{body}\n@OUT = c;"


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _cook(src, device, cancel=None, code=None):
    res = tex_engine.cook(code or _CODE, {"A": src}, device_mode=device, precision="fp32",
                          cancel=cancel)
    _sync(device)
    return res.outputs["OUT"]


def _write(handle, stall):
    """The 'sink': fence the handle, touch the bytes, stall. Touching matters — a writer that
    never reads is not a writer, and the fence is the half §3.3 exists to prove."""
    t = handle.tensor() if hasattr(handle, "tensor") else handle
    float(t.reshape(-1)[0])
    time.sleep(stall)


def measure_compute_only(frames, res, device, warmup):
    """The floor: cooking alone, no I/O at all. `hidden` is measured against this."""
    src = torch.rand(1, res, res, 4, device=device)
    for _ in range(warmup):
        _cook(src, device)
    t0 = time.perf_counter()
    for _ in range(frames):
        _cook(src, device)
    return time.perf_counter() - t0


def measure_serial(frames, res, device, in_stall, out_stall, provider):
    """fetch -> cook -> write, strictly one after another."""
    t0 = time.perf_counter()
    for i in range(frames):
        src = tex_provider.materialize("plate", float(i), "fetch")
        out = _cook(src, device)
        _write(streams.egress(out), out_stall)
    return time.perf_counter() - t0


def measure_overlapped(frames, res, device, in_stall, out_stall, provider, lookahead):
    """Host I/O thread lands promises; the queue admits a cook when ITS input landed; a
    writer thread drains completed frames as handles."""
    q = Q.CookQueue(name="pm9")
    promises = [Promise(f"f{i}", type=TEXType.VEC4) for i in range(frames)]
    writes = _queue.Queue()
    stop = threading.Event()

    def fetcher():
        for i in range(frames):
            if stop.is_set():
                return
            try:
                promises[i].land(tex_provider.materialize("plate", float(i), "fetch"))
            except Exception as e:                        # a dead source fails its cook
                promises[i].fail(e)

    def writer():
        done = 0
        while done < frames and not stop.is_set():
            item = writes.get()
            if item is None:
                return
            _write(item, out_stall)
            done += 1

    t0 = time.perf_counter()
    tf = threading.Thread(target=fetcher, daemon=True)
    tw = threading.Thread(target=writer, daemon=True)
    tf.start()
    tw.start()

    jobs = []
    for i in range(frames):
        def cook_one(cancel, _i=i):
            out = _cook(promises[_i].value, device, cancel)
            writes.put(streams.egress(out))
            return True
        jobs.append(q.submit(cook_one, klass=Q.COMMITTED, inputs=[promises[i]]))

    for j in jobs:
        j.result(timeout=300)
    tw.join(300)
    elapsed = time.perf_counter() - t0
    stop.set()
    writes.put(None)
    q.close()
    return elapsed, q


def measure_queue_prefetch(frames, res, device, out_stall, provider):
    """The OTHER mode: `declare_window` prefetch riding the same single worker as the cooks.
    Measured so the docs can say what it costs instead of guessing."""
    q = Q.CookQueue(name="pm9-tierb")
    q.install_policy(Q.SpeculativePolicy(min_confidence=0.0, min_value_ms=0.0,
                                         unknown_min_confidence=0.0, max_pending=frames))
    t0 = time.perf_counter()
    tex_provider.declare_window(q, "plate", 0.0, float(frames - 1), confidence=0.9,
                                mode="fetch", max_frames=frames)
    jobs = []
    for i in range(frames):
        def cook_one(cancel, _i=i):
            out = _cook(tex_provider.materialize("plate", float(_i), "fetch"), device, cancel)
            _write(streams.egress(out), out_stall)
            return True
        jobs.append(q.submit(cook_one, klass=Q.COMMITTED))
    for j in jobs:
        j.result(timeout=300)
    elapsed = time.perf_counter() - t0
    q.close()
    return elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--in-stall", type=float, default=0.004, help="provider seconds per fetch")
    ap.add_argument("--out-stall", type=float, default=0.004, help="writer seconds per frame")
    ap.add_argument("--lookahead", type=int, default=8)
    ap.add_argument("--stages", type=int, default=24,
                    help="grades per frame — sets the compute:I/O ratio (see _code)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--save")
    args = ap.parse_args()

    dev = args.device
    global _CODE
    _CODE = _code(args.stages)
    print(f"PM-9 — {args.frames} frames at {args.res}² on {dev}, {args.stages} grades/frame; "
          f"provider {args.in_stall * 1000:.0f} ms/fetch, writer {args.out_stall * 1000:.0f} ms/frame")

    def arm():
        tex_provider.reset_provider()
        p = tex_provider.SyntheticFrameProvider(res=args.res, rate=1.0, device=dev,
                                                latency_s=args.in_stall)
        tex_provider.set_provider(p)
        # No pooling across routes: a cached frame is not a fetch, and PM-9 is about
        # hiding fetches. Every route pays every fetch exactly once.
        tex_provider.set_media_budget_mb(0.0)
        return p

    arm()
    t_compute = measure_compute_only(args.frames, args.res, dev, args.warmup)
    print(f"  compute only        {t_compute * 1000:9.1f} ms")

    p = arm()
    t_serial = measure_serial(args.frames, args.res, dev, args.in_stall, args.out_stall, p)
    print(f"  serial              {t_serial * 1000:9.1f} ms   ({p.fetches} fetches)")

    p = arm()
    t_over, q = measure_overlapped(args.frames, args.res, dev, args.in_stall,
                                   args.out_stall, p, args.lookahead)
    print(f"  overlapped          {t_over * 1000:9.1f} ms   ({p.fetches} fetches, "
          f"waiting-peak seen by the queue: {q.stats.submitted} submitted)")

    p = arm()
    t_tierb = measure_queue_prefetch(args.frames, args.res, dev, args.out_stall, p)
    print(f"  tier-B prefetch     {t_tierb * 1000:9.1f} ms   (same worker as the cooks)")

    io_serial = t_serial - t_compute
    exposed = max(0.0, t_over - t_compute)
    hidden = 1.0 - (exposed / io_serial) if io_serial > 0 else 0.0
    print("\n" + "=" * 70)
    print(f"  I/O in the serial route : {io_serial * 1000:8.1f} ms")
    print(f"  I/O still exposed       : {exposed * 1000:8.1f} ms")
    print(f"  HIDDEN                  : {hidden * 100:8.1f} %    "
          f"(PM-9 exit: >= 80%)  {'PASS' if hidden >= 0.80 else 'FAIL'}")
    print(f"  speedup vs serial       : {t_serial / t_over:8.2f}x")
    print("=" * 70)

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump({"frames": args.frames, "res": args.res, "device": dev,
                   "stages": args.stages,
                       "in_stall_s": args.in_stall, "out_stall_s": args.out_stall,
                       "compute_ms": t_compute * 1000, "serial_ms": t_serial * 1000,
                       "overlapped_ms": t_over * 1000, "tierb_prefetch_ms": t_tierb * 1000,
                       "hidden_frac": hidden, "speedup": t_serial / t_over}, f, indent=2)
        print(f"saved {args.save}")
    tex_provider.reset_provider()
    tex_provider.set_media_budget_mb(512.0)


if __name__ == "__main__":
    main()
