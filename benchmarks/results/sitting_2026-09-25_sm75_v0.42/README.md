# Release sitting — 2026-09-25, sm_75 reference box

The timing record for **v0.42.2**, taken against `base` = **v0.40.3**, under the project's
standing sitting rules. This directory is that sitting: every comparison read off it, and the
commands that reproduce it.

`.comfyignore` excludes `benchmarks/`, so nothing here reaches the published registry archive.
The raw per-leg result JSON (48 files) is **not** in this directory or in the repository: it
lives outside the repository, per the rule that a sitting commits its reading, not its raw legs.

## The box

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 2080 SUPER (**sm_75**) — the reference box, a desktop |
| torch / python | 2.12.0+cu130 / 3.11.9 |
| state | quiet: 0 other python processes and an idle GPU checked before the sitting started and between every leg; nothing else ran on the box for the sitting's duration |

Not the sm_120 development laptop, for the same reason as every prior sitting: its clocks
idle-collapse under mixed load. The device-independent counter rows in the project's structural
counts harness are the property that lets a timing sitting run on hardware the laptop's own
benchmarks were never taken on — re-proofed below.

## The legs

Eight legs, each running the same six benches (five timing benches plus the CUDA-shaped counts
harness) in the same order, each with **its own cache directory, emptied before the leg started**
(the same runner and comparator used for the prior sittings in this series, unmodified).

| leg | tree | role |
|---|---|---|
| `after_discard` | `v0.42.2` | **discarded** — first run of the freshly-materialised `after` tree |
| `base_a1` | `v0.40.3` | base — with `base_a2`, the opening **null control** |
| `base_a2` | `v0.40.3` | base — opening null control |
| `after_b` | `v0.42.2` | after, round 1 |
| `base_a3` | `v0.40.3` | base — reference for round 1's read and round 2's |
| `after_c` | `v0.42.2` | after, round 2 |
| `base_a4` | `v0.40.3` | base — reference for round 2's read and the closing null |
| `base_a5` | `v0.40.3` | base — the **closing null control** against `base_a4` |

Both trees were shipped fresh immediately before the sitting; the `after` sha was confirmed a
descendant of the `base` sha beforehand. All eight legs ran back to back, unattended, in one
sequence, with a per-leg preflight/postflight process-and-GPU check; every leg's preflight found 0
other python processes, and every one of the 48 bench invocations returned `rc=0`. Total wall time
for the sequence: ~5h45m.

## The comparisons

Each `read_*.txt` is the output of the comparator over the legs named in its first line. Ratios
are `reference / leg`, so **>1 means the leg is faster**.

| file | legs | what it answers |
|---|---|---|
| `read_null_a1_a2.txt` | `base_a1`, `base_a2` | the opening **null control**: identical code against itself |
| `read_after_b.txt` | `base_a2`, `after_b`, `base_a3` | round 1: base → after → base |
| `read_after_c.txt` | `base_a3`, `after_c`, `base_a4` | round 2: base → after → base |
| `read_null_a4_a5.txt` | `base_a4`, `base_a5` | the closing **null control**, in the same sequence as round 2's claim |

## The null band, and how to read the result

**eight_config @512² per-config geomeans** (the interpretable unit; individual rows are noise):

| config | opening null (`a2`/`a1`) | round 1 `after_b`/`a2` | round 1 null `a3`/`a2` | round 2 `after_c`/`a3` | round 2 null `a4`/`a3` | closing null (`a5`/`a4`) |
|---|---|---|---|---|---|---|
| cpu_off_cold | 0.986 | 1.041 | 1.017 | 1.023 | 0.976 | 1.032 |
| cpu_off_warm | 0.991 | 0.972 | 1.004 | 1.004 | 1.005 | 0.987 |
| **cuda_off_cold** | **0.985** | **1.063** | **1.000** | **1.066** | **1.006** | **0.987** |
| cuda_off_warm | 0.992 | 1.000 | 0.991 | 1.011 | 1.008 | 0.999 |
| cpu_on_cold | 1.053 | 0.998 | 1.002 | 0.995 | 0.989 | 1.007 |
| cpu_on_warm | 0.998 | 0.990 | 1.005 | 1.002 | 0.987 | 1.005 |
| cuda_on_cold | 1.000 | 1.014 | 1.007 | 1.008 | 1.007 | 0.995 |
| cuda_on_warm | 1.000 | 1.010 | 1.008 | 1.007 | 1.002 | 0.993 |

**`cuda_off_cold` — the interpreter backend's first CUDA touch — moves outside the null band in
both after legs, in the improving direction.** The four null readings bracketing the two after
legs sit at 0.985, 1.000, 1.006 and 0.987 — all within about ±1.5% of 1.00. Both after legs read
well above that: **1.063** and **1.066**, reproduced independently twice, on a metric this
series has watched closely: a previous release in this line read this same config at 0.949–0.950
against the same ~0.99–1.02 null band, a regression that was traced and fixed afterwards. This
sitting's reading is the fix showing up in the same measurement: not merely "no longer
regressed" but reading faster than the pre-regression baseline, by an amount consistent with an
independent same-box re-measurement of the fix taken outside this sitting. No other config in the
corpus, including `cuda_on_cold` (the compiled path's own cold reading), moves by more than ~1.5%
beyond its own bracketing nulls in either after leg.

**Every other eight_config geomean reads inside or close to its own round's null band.** The
default whole-frame cook path — the thing invariant to preserve across a release — is neutral,
matching the release's own claim that the corpus does not exercise its new surface.

**The four benches that isolate an interactive cost** mostly read neutral, with one exception
worth naming precisely because it moved consistently and the null legs did not: `roi_scrub`'s
`roi_pan_param` row (a panning ROI window combined with a scrubbing parameter) reads **13.4–13.9%
faster** in both after legs (round 1: 1.139, round 2: 1.134) against null-leg readings of 0.998
and 0.997 for the same row in the same rounds — a real, reproduced, per-row speedup, not a
per-config regression. `region_recook`, `param_scrub`'s recook/scrub/static medians, and
`cookqueue`'s interactive rows otherwise stayed within the swings their own null legs showed in
the same rounds (`cookqueue`'s `queued_idle_ms` and `param_scrub`'s worst-tick-after-warm-up
remain this box's known noisy metrics, moving by similar amounts in both after and null legs).

**The structural counts harness shows one thing outside the default path that the timing rows
don't capture on their own.** Comparing `after` against its bracketing `base` leg, 0 rows moved in
the opening and closing null pairs (identical code against itself), but **the same 7 counter rows
moved identically in both independent after legs**, all in interactive re-cook scenarios rather
than the default whole-frame path: a source-edit re-cook, a parameter-pan re-cook, an
all-dirty re-cook, and a prewarm pass each now perform fewer cache-fingerprint calls, fewer tile
plan re-derivations, fewer allocations, or fewer kernel launches than the same scenario on `base`.
Because this is counter-backed (not a timing artifact) and reproduces exactly across two
independent legs while reading 0/0 in both null legs, it is recorded as a genuine, structural
difference between the two trees in these scenarios — read as a reduction in redundant work on
interactive re-cook paths, not a correctness concern, but outside what the eight_config default
path alone would show.

## Verdict

**The default whole-frame per-cook path reads neutral.** One config moves clearly outside the
null band in both after legs: `cuda_off_cold`, reading 1.063–1.066 against a 0.985–1.006 null
band — an improvement, consistent with a fix to this same metric that this project's own
bisection work had already isolated and attributed to the interpreter backend's first-CUDA-touch
setup cost. One interactive-bench row (`roi_scrub`'s `roi_pan_param`) reads a reproduced,
null-exceeding speedup (~13.6%). The structural counts harness additionally shows 7 counter rows
moving identically in both after legs, all reductions in interactive re-cook work, absent from
both null controls — flagged here as a real, counter-backed finding for the release record, not
a per-cook regression on the default path.

## Counts re-proof

The project's structural counts test, run against the `after` tree's parent directory on the
reference box (CUDA present, so the CUDA-pinned row executed rather than skipping):

```
.......                                                                  [100%]
7 passed, 1 warning in 5.3s
```

`rc=0`. The one warning is torch's own profiler notice about clearing events between cycles,
unrelated to the pins. (A `triton`-import stack trace and a `logging` teardown error also appear
in the raw output, both after the `7 passed` line and both artifacts of `torch.compile` probing
for an unavailable Triton backend during interpreter shutdown on this box — harmless, the same
class of shutdown noise seen in prior sittings, and do not affect the reported result.) All seven
pins, including the CUDA-shaped row, matched: the device-independent counter rows this release's
`after` tree produces on **sm_75** agree with the values pinned against the development
hardware's **sm_120**. No row differs between the two boxes; no escalation from this half of the
re-proof.

## Reproducing it

Unchanged from the prior sittings in this series: the same runner and comparator, reused as-is,
per-leg cache directories emptied first, the first leg of every freshly-materialised tree
discarded, and read with the comparator over the results directory and the legs to compare.
