# LAT-1b — asynchronous CUDA-graph capture (mini-design)

*v0.23. The "hard half" of LAT-1 (LAT-1a — background `torch.compile` — was itself
deferred in v0.21: `torch.compile` is lazy, so a background wrap doesn't hide the first
*execution* stall). This is LAT-1b's own design note, in the xpu-transfer-scheduling.md
mold: the shipped state, the design, the crux that makes it hard, the measurement, and
the gate that would flip the decision.*

## Shipped state (v0.23)

CUDA-graph capture is **synchronous, in-cook**. On the first cook of a capturable program
at a given (fingerprint × device × precision × shape) key, `graphed.run_graphed` builds
static staging buffers, warms up (3 runs on a side stream), and captures under
`torch.cuda.graph(capture_error_mode="thread_local")` — all on the cook thread, before it
can return (`graphed._capture_inner`). Every subsequent cook of that key replays.

**Measured cost** (sm_120, torch 2.12, a 2-builtin program at 1024²):

| | ms |
|---|---|
| first cook (warmup ×3 + capture + syncs) | **72.1** |
| warm replay (steady state) | 0.34 |
| interpreter, same program warm | 0.34 |

So the stall is a **one-time ~72 ms**, and — this is the load-bearing fact —
**the capture key excludes `$param` values** (`_capture_key`: fingerprint + shapes +
device + precision, never param values). An interactive scrub that twiddles a knob keeps
one key, so it captures **once** and every later frame replays at 0.34 ms. The 72 ms is
paid once per *(program, resolution)*, never per frame.

## The async design (what LAT-1b would build)

Mirror the `auto` tier's serve-fast-swap-on-ready state machine (`compiled.run_auto` +
`autotier`, the pattern LAT-1a's forced path was meant to reuse):

1. First sight of a capturable key: **do not capture in-cook**. Serve the frame via
   codegen/interpreter (no stall) and submit the capture to a worker.
2. The worker warms up + captures on the cook's device, then publishes the
   `GraphedProgram` into `_graph_cache`.
3. Later cooks: replay if the graph is ready, else keep serving codegen; blacklist on
   capture failure. This is exactly the `MEASURING → COMPILING → COMMITTED/REJECTED`
   shape `autotier` already runs, with "capture" in place of "compile".

The verdict/blacklist persistence (`autotier.json` idiom) would let a program's
"graphable" decision survive a restart (CACHE-3 territory) so the first scrub after a
relaunch doesn't re-pay discovery.

## The crux (why (b) "must not ride (a)'s coattails")

`torch.compile`'s background compile is mostly host-side; the worker touching the device
concurrently with a foreground cook is tolerated (the auto tier only gates it on memory
headroom + `_capture_in_flight`). **CUDA-graph capture is categorically different: while a
stream is capturing, the CUDA context forbids concurrent work** — a foreground cook (of
*any* program, not just graph cooks) issuing to the device during capture corrupts the
capture or the allocator. `capture_error_mode="thread_local"` scopes only *capture-illegal
ops on the capturing thread*; it does **not** make concurrent device work from another
thread safe.

So async capture needs the capture worker to hold the device **exclusively** for the
capture window. Under ComfyUI's one-cook-at-a-time executor the device is idle *between*
cooks — the natural capture window — but there is no clean "device idle" signal, so the
worker must **serialize against every foreground device cook**, not just graph ones. That
is a lock (or an idle-callback) threaded through the engine's whole cook path (an S-1
boundary), plus:

- device-pinned capture/replay and RNG-poison recovery on the worker's device (HW-2),
- the per-graph allocator pool and `_build_keepalive`/`_CAPTURING` invariants (AGENTS §91,
  a DO-NOT-TOUCH register entry),
- MEM-1 stale-address safety when a captured pool is later evicted,
- a **GPU soak test** proving the serialization is free of deadlock, allocator corruption,
  and — the silent-wrong risk — a graph captured against a perturbed device state.

The coordination seed already exists: the process-wide `_CAPTURING` flag and
`graphed.is_capturing()`, which the auto tier reads to avoid background-compiling during a
capture. LAT-1b generalizes that from "don't compile now" to "don't cook now," which is
the invasive part.

## Decision: DEFER the implementation; land this design

The measured stall is **one-time per key and off the steady-state path** (replay is
0.34 ms, identical to the interpreter here — the win-region gate `_graph_capture_worthwhile`
already keeps capture from firing where it wouldn't pay). Weighed against building a
whole-cook device-serialization lock through the engine seam and soak-validating it against
a DO-NOT-TOUCH tier, the async path is **not justified by the current measurement** — the
same call the roadmap made for LAT-1a and LAT-2, on the same evidence shape (a real but
one-time or small cost next to a large validation surface).

**The gate that flips this decision** — reopen LAT-1b when a measurement shows *repeated*
capture stalls on an interactive path, i.e. the one-time assumption breaks. Concretely:
an interactive session that churns capture keys (resolution scrubbing, or LANG-6 canvas
changes, or many distinct programs in a graph) so the 72 ms is paid often rather than
once. At that point the serve-codegen-swap-on-ready machine above earns its coordination
cost, and the persistence idiom (CACHE-3) removes the relaunch re-pay.

**Not measured, so not claimed:** the stall at larger resolutions (capture cost grows with
the kernel count baked in) and on multi-GPU (each device index is its own key, so a
two-GPU interactive session pays capture per device). Both would be inputs to the reopen.
