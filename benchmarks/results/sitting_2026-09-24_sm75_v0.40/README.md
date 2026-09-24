# Release sitting — 2026-09-24, sm_75 reference box

The timing record for **v0.40.3 "Close the ledger"**, taken against `base` = **v0.39.0**
(`256c81d`) under the rules in `docs/roadmap.md` §10 item 3. This directory is that sitting:
every comparison read off it, and the commands that reproduce it.

`.comfyignore` excludes `benchmarks/`, so nothing here reaches the published registry archive.
The raw per-leg result JSON (48 files) is **not** in this directory or in the repository: it
lives outside the repository, with the maintainer, per the rule that a sitting commits its
reading, not its raw legs.

## The box

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 2080 SUPER (**sm_75**) — the reference box, a desktop |
| torch / python | 2.12.0+cu130 / 3.11.9 |
| state | quiet: 0 other python processes and an idle GPU checked before the sitting started and between every leg; nothing else ran on the box for the sitting's duration |

Not the sm_120 development laptop, for the same reason as every prior sitting: its clocks
idle-collapse under mixed load. The device-independent counter rows in
`tests/test_bench2_counts.py` are the property that lets a timing sitting run on hardware the
laptop's own benchmarks were never taken on — re-proofed below.

## The legs

Eight legs, each running the same six benches (five timing benches plus the CUDA-shaped counts
harness) in the same order, each with **its own cache directory, emptied before the leg started**
(the same runner and comparator used for the two prior sittings, unmodified).

| leg | tree | role |
|---|---|---|
| `after_discard` | `v0.40.3` | **discarded** — first run of the freshly-materialised `after` tree |
| `base_a1` | `v0.39.0` | base — with `base_a2`, the opening **null control** |
| `base_a2` | `v0.39.0` | base — opening null control |
| `after_b` | `v0.40.3` | after, round 1 |
| `base_a3` | `v0.39.0` | base — reference for round 1's read and round 2's |
| `after_c` | `v0.40.3` | after, round 2 |
| `base_a4` | `v0.39.0` | base — reference for round 2's read and the closing null |
| `base_a5` | `v0.39.0` | base — the **closing null control** against `base_a4` |

Both trees were shipped fresh immediately before the sitting; the `after` sha was confirmed a
descendant of the `base` sha beforehand. All eight legs ran back to back, unattended, in one
sequence, with a per-leg preflight/postflight process-and-GPU check; every leg's preflight found 0
other python processes, and every one of the 48 bench invocations returned `rc=0`. One connection
hiccup interrupted the controlling session partway through (leg 6 of 8), but the sitting itself
kept running unaffected on the box and was polled back to completion rather than restarted or
touched. Total wall time for the sequence: ~5h40m.

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
| cpu_off_cold | 1.006 | 0.962 | 0.982 | 0.979 | 1.003 | 1.023 |
| cpu_off_warm | 0.983 | 0.987 | 1.011 | 0.979 | 0.995 | 0.971 |
| **cuda_off_cold** | **1.003** | **0.949** | **0.990** | **0.950** | **1.008** | **1.015** |
| cuda_off_warm | 0.998 | 1.000 | 0.986 | 1.010 | 1.008 | 1.017 |
| cpu_on_cold | 1.014 | 0.982 | 0.990 | 0.995 | 0.993 | 1.006 |
| cpu_on_warm | 0.992 | 0.996 | 0.997 | 0.994 | 1.006 | 1.000 |
| cuda_on_cold | 1.002 | 0.994 | 1.002 | 0.993 | 0.998 | 0.999 |
| cuda_on_warm | 0.995 | 0.997 | 1.006 | 0.987 | 0.998 | 0.995 |

**One config moved outside its null band, twice, by the same amount: `cuda_off_cold`.** Both
after legs read **0.949** and **0.950** — the interpreter path's first touch of CUDA, on the
freshly-materialised tree's cold cache. Every one of the four null readings bracketing them
(opening 1.003, round 1's `a3` 0.990, round 2's `a4` 1.008, closing 1.015) sits at or above 0.99,
and no other config — including `cuda_on_cold`, the compiled path's own cold reading — moves by
more than 1.3% in either after leg. That is the signature the sitting's own methodology is built
to catch: the same effect, reproduced independently twice, absent from four same-box, same-tree
null measurements taken in the same sequence. **This sits at the 0.95 stop-ship line.**

Read against `counts_cuda`: **0 api/cuda rows moved** in either after leg (23 frame rows moved in
both — consistent between the two legs, the expected signature of new call sites added by lawful
change, not gating). The steady-state per-cook counters — the actual release gate — see nothing,
which means this is not a per-cook cost: the counts harness measures repeating steady ticks and
would not see a cost paid once per cold process. A ~5% cold-only cost isolated to the interpreter
backend's first CUDA touch is consistent in shape with new one-time setup work on that path —
this release adds CUDA-graph capture for the new viewer builtins, plus the new colour builtins,
`apply_lut3d`, and masked-flow call sites, any of which could add first-touch registration or
context-setup cost without touching the per-tick call count. **Flagging this as the sitting's
finding, not confirming a root cause**: it clears the same-sequence null-control bar the
methodology sets, but no counter row currently explains it, and the roadmap's own rule is that a
claim like this is named by a counter before it is named by a ratio — that naming has not
happened yet. Worth a follow-up before or shortly after release: a counter for first-CUDA-touch
setup cost, or confirmation that the ~5% is accepted as a one-time process cost distinct from the
per-cook path this release promises is unchanged.

Every other config's after-leg reading sits inside or close to the range its own null control
established in the same round (e.g. `cpu_off_cold` 0.962/0.979 against nulls 0.982/1.003 — a
softer, ~2% echo on the CPU-interpreter side, well inside this box's known wide CPU noise band
and not treated as a separate finding).

**The four benches that isolate an interactive cost** read neutral in both rounds: `roi_scrub`
and `region_recook`'s CUDA rows stayed within 0.6% of their null neighbours in both after legs;
`param_scrub`'s recook median and `cookqueue`'s three interactive rows moved less in the after
legs than the null legs moved against each other in the same rounds (e.g. round 1's null showed
`cookqueue preempt_to_first_stmt` at 0.868 and `queued_under_load` at 0.914 — both wider swings
than either after-leg row). One single-row phantom is worth naming so a reader doesn't chase it:
`region_recook cpu/n50/2048/whole_all` read 1037 ms on `base_a3` against 865–910 ms on every
other leg of the sitting including the same tree immediately before and after — the same
first-run-style single-row spike this box has produced before, on a leg that was not itself a
tree's first run. It inflates the raw ratios quoted "against `a3`" in `read_after_c.txt` for that
one row; it is not an after-leg effect.

## Verdict

**The default per-cook path is neutral — counts confirm it (0 api/cuda rows moved in both after
legs), matching the release's own claim that no bench in this corpus exercises the new surface.**
One finding outside that: `cuda_off_cold`'s geomean sits at 0.949–0.950 in both after legs against
a 0.99–1.02 null band across all four null readings — reproduced, at the stop-ship line, and not
yet explained by a counter. Recorded here as the sitting's escalation candidate for the release
owner to name a counter against or accept as a one-time cost; not treated as a per-cook regression
because the steady-state counts do not move.

## Counts re-proof

`pytest tests/test_bench2_counts.py -q -p no:cacheprovider`, run against the `after` tree's parent
directory on the reference box (CUDA present, so the CUDA-pinned row executed rather than
skipping):

```
.......                                                                  [100%]
7 passed, 1 warning in 5.03s
```

`rc=0`. The one warning is torch's own profiler notice about clearing events between cycles,
unrelated to the pins. (A `triton`-import stack trace and a `logging` teardown error also appear
in the raw output, both after the `7 passed` line and both artifacts of `torch.compile` probing
for an unavailable Triton backend during interpreter shutdown on this box — harmless, the same
class of shutdown noise seen in the prior sitting, and do not affect the reported result.) All
seven pins, including the CUDA-shaped row, matched on first run: the device-independent counter
rows this release's `after` tree produces on **sm_75** agree with the values pinned against the
development laptop's **sm_120**. No row differs between the two boxes; no escalation from this
half of the re-proof.

## Reproducing it

Unchanged from the two prior sittings in this directory: the same runner and comparator, reused
as-is, per-leg cache directories emptied first, the first leg of every freshly-materialised tree
discarded, and read with the comparator over the results directory and the legs to compare.
