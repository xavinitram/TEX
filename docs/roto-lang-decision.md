# ROTO-lang — decided on a measurement: **not shipped**

*Doc 42 §2.5 / doc 40 §4.6. Four releases overdue, and this is the last window before DATA-5's
design doc opens at v0.37 — the gate was "before, not after". Either verdict closes the item;
this one is a no-go, recorded with the number that produced it.*

---

## 1. The question

ROTO-lang would add `sdf_bezier(px, py, p0..p3)` and `spline_mask(px, py, points, feather)` to
the stdlib, so a roto shape becomes a **fusable procedural mask** — evaluated per pixel inside
the comp rather than rasterized by the host and wired in as a MASK plane.

It has always been conditional. Doc 40 §4.6: *"ships only if the app's measurements say fusable
procedural masks beat host-rasterized MASK planes."* Nobody had measured. This does.

## 2. What was measured

`benchmarks/roto_spike.py`, sm_75 / RTX 2080 SUPER, medians of 5 runs with
`torch.cuda.synchronize()` around every timed CUDA region. An 8-segment closed polygon — the
shape a roto user actually draws — with per-segment distance and an even-odd winding test.

* **procedural** — one TEX cook evaluating the SDF per pixel and applying it. The distance and
  winding arithmetic is unrolled, so the interpreter's loop overhead is *not* in the number;
  this flatters the route that loses.
* **rasterized** — the host builds the same mask once with plain torch ops, then a TEX cook
  applies it as an ordinary MASK binding.

Both routes produce the same **hard-edged** mask, so the comparison is about cost alone. (The
first cut of the spike wrote the mask as `inside*(1-smoothstep(..)) + inside*smoothstep(..)`,
which is algebraically just `inside` — two smoothsteps per pixel charged to the procedural
column for a value it already had, and a `$feather` the rasterized side never applied. Caught
by a /simplify pass and re-measured; the numbers below are the corrected run. It moved the
procedural column by under 1%, which is its own small finding: against an 8-segment per-pixel
SDF, two smoothsteps are noise.)

| res | device | procedural | rasterize once | apply | first frame | unrelated scrub |
|---|---|---|---|---|---|---|
| 1080² | cpu | 107.83 ms | 10.95 ms | 1.16 ms | 8.90× | **93.18×** |
| 2160² | cpu | 368.22 ms | 66.46 ms | 15.58 ms | 4.49× | **23.64×** |
| 1080² | cuda | 13.47 ms | 5.03 ms | 0.75 ms | 2.33× | **17.88×** |
| 2160² | cuda | 49.75 ms | 18.43 ms | 2.54 ms | 2.37× | **19.57×** |

Geomeans — procedural ÷ rasterized, so >1 means procedural is slower:
**first frame 6.32× (cpu) / 2.35× (cuda); unrelated scrub 46.93× (cpu) / 18.70× (cuda).**

## 3. The verdict, and the mechanism

**Not shipped.** Procedural masks lose on both axes, on both devices, at both resolutions.

The first-frame column is the *charitable* comparison and procedural still loses 2.3–8.9×,
because a mask is cheap to rasterize and a per-pixel SDF over 8 segments is not: the host
evaluates the same arithmetic once into a `[H,W]` buffer with vectorised torch ops, while the
program re-derives it inside a `[B,H,W,C]` expression tree.

The scrub column is the one that decides it, and it is not close — **18× to 93×**. A roto shape
changes rarely; the things a user drags continuously are grade and blur parameters. A
rasterized mask is a *host-side value* that survives those edits untouched, so the cook applies
a ready buffer. A procedural mask is part of the program, so ANIM-1's own guarantee works
against it here: changing `$exposure` re-cooks the program, and the program contains the mask.
The feature's headline benefit — that the mask fuses with the comp — is exactly what denies it
the caching every other part of the pipeline gets.

Fusion cannot rescue this either. v0.34's FUS-cap spike measured that fusing a long chain at
production resolution buys nothing on CUDA and costs 12.9× peak memory
(`docs/fusion-cap-decision.md`); a mask stage folded into a fused region inherits that, and
still recomputes per cook.

## 4. What this closes, and what it does not

* **ROTO-lang is closed before it cost any stdlib surface** — no `sdf_bezier`, no
  `spline_mask`, no iterative-Newton convergence proof across degenerate control points, no
  FP16_FRAGILE classification, no fuzzer exclusion. Doc 40 §4.6 counts a recorded no-go as a
  completed item, and the cost avoided is the point.
* **DATA-5's shape is unblocked.** The gate existed because DATA-5's border policies and
  realization templates were designed against an answer to this. The answer is: masks arrive as
  planes, so DATA-5 plans for MASK-plane inputs and needs no spline surface.
* **Roto tooling is host territory, as the standing register already said.** Interaction,
  stroke capture, AI assists and the rasterizer itself were never engine items; this measurement
  says the *evaluation* belongs there too.
* **Not closed: whether TEX should read a mask more cheaply.** `apply` is 0.75–15.58 ms, most of
  it the ordinary cost of a full-frame multiply. DATA-6's plane bindings (v0.35) are the
  relevant lever — a mask arriving as a named plane rather than a separate wire — and that is
  already scheduled.

## 5. The reopen gate

One measurement reopens this: **a mask whose rasterized form cannot be reused.** If a host ever
drives roto points as a per-frame animated parameter — a tracked shape following a moving
object — then the rasterized mask is rebuilt every frame too, the scrub column collapses to the
first-frame column, and 2.3× on CUDA is close enough to re-argue. That is a real workflow
(planar tracking), it is simply not the one the numbers above describe, and it would need its
own harness driving the point array per frame rather than holding it fixed.

Secondary: a kernel-level fusion backend (GRAPH-2, or a working Triton path) changes the
procedural route's cost model the same way it changes FUS-cap's. Re-measure the day one exists.

*sm_75, the standing hardware caveat. Raw rows: `benchmarks/roto_spike.py --save`.*
