# Release sitting — 2026-09-23/24, sm_75 reference box

The timing record for **v0.39.0 "Every pixel its own way"**, taken against `base` = **v0.38.0**
(`27160bd`) under the rules in `docs/roadmap.md` §10 item 3. This directory is that sitting:
every comparison read off it, and the commands that reproduce it. It supersedes the placeholder
of the same name written by an earlier attempt that found the reference box unreachable; the box
came back and this sitting ran to completion on it in one unattended sequence.

`.comfyignore` excludes `benchmarks/`, so nothing here reaches the published registry archive.
The raw per-leg result JSON (48 files) is **not** in this directory or in the repository: it sits
outside the repository with the maintainer and on the reference box itself, per the
rule that a sitting commits its reading, not its raw legs.

## The box

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 2080 SUPER (**sm_75**) — the reference box, a desktop |
| torch / python | 2.12.0+cu130 / 3.11.9 |
| state | quiet: 0 other python processes and GPU idling at 1650 MHz (P0) checked before the sitting started and between every leg; nothing else ran on the box for the sitting's duration |

Not the sm_120 development laptop, for the same reason as every prior sitting: its clocks
idle-collapse under mixed load. The device-independent counter rows in
`tests/test_bench2_counts.py` are the property that lets a timing sitting run on hardware the
laptop's own benchmarks were never taken on — re-proofed below.

## The legs

Eight legs, each running the same six benches (five timing benches plus the CUDA-shaped counts
harness) in the same order, each with **its own cache directory, emptied before the leg started**
(`sitting.ps1`, unmodified from the v0.38.0 sitting).

| leg | tree | role |
|---|---|---|
| `after_discard` | `v0.39.0` (`256c81d`) | **discarded** — first run of the freshly-materialised `after` tree |
| `base_a1` | `v0.38.0` (`27160bd`) | base — with `base_a2`, the opening **null control** |
| `base_a2` | `v0.38.0` | base — opening null control |
| `after_b` | `v0.39.0` | after, round 1 |
| `base_a3` | `v0.38.0` | base — reference for round 1's read and round 2's |
| `after_c` | `v0.39.0` | after, round 2 |
| `base_a4` | `v0.38.0` | base — reference for round 2's read and the closing null |
| `base_a5` | `v0.38.0` | base — the **closing null control** against `base_a4` |

Both trees were shipped fresh with `git archive <sha> | ssh ... tar -xf -` immediately before the
sitting (`27160bd` for `base`, `256c81d` for `after`); `27160bd` was confirmed an ancestor of
`256c81d` beforehand. All eight legs ran back to back, unattended, in one sequence, with a
per-leg preflight/postflight process-and-GPU check baked into `sitting.ps1`; every leg's
preflight found 0 other python processes, and every one of the 48 bench invocations returned
`rc=0`. Total wall time for the sequence: ~5h35m.

## The comparisons

Each `read_*.txt` is the output of `sitting_compare.py` over the legs named in its first line.
Ratios are `reference / leg`, so **>1 means the leg is faster**.

| file | legs | what it answers |
|---|---|---|
| `read_null_a1_a2.txt` | `base_a1`, `base_a2` | the opening **null control**: identical code against itself |
| `read_after_b.txt` | `base_a2`, `after_b`, `base_a3` | round 1: base → after → base |
| `read_after_c.txt` | `base_a3`, `after_c`, `base_a4` | round 2: base → after → base |
| `read_null_a4_a5.txt` | `base_a4`, `base_a5` | the closing **null control**, in the same sequence as round 2's claim |

## The null band, and how to read the result

**eight_config @512² per-config geomeans** (the interpretable unit; individual rows are noise —
row ranges below are for context only):

| config | opening null (`a2` vs `a1`) | round 1 `after_b` vs `a2` | round 1 null `a3` vs `a2` | round 2 `after_c` vs `a3` | round 2 null `a4` vs `a3` | closing null (`a5` vs `a4`) |
|---|---|---|---|---|---|---|
| cpu_off_cold | 1.011 | 0.979 | 0.988 | 0.997 | 1.004 | 1.007 |
| cpu_off_warm | 0.992 | 0.991 | 0.984 | 1.028 | 1.013 | 0.995 |
| cuda_off_cold | 1.001 | 0.995 | 1.011 | 0.984 | 1.003 | 0.989 |
| cuda_off_warm | 0.994 | 1.000 | 1.010 | 0.987 | 0.985 | 1.006 |
| cpu_on_cold | 1.046 | 0.995 | 0.999 | 0.999 | 0.993 | 1.005 |
| cpu_on_warm | 0.996 | 0.988 | 1.000 | 1.006 | 0.982 | 1.015 |
| cuda_on_cold | 1.003 | 0.986 | 0.997 | 0.984 | 0.997 | 1.003 |
| cuda_on_warm | 1.002 | 0.990 | 0.998 | 0.993 | 1.002 | 0.998 |

**`after_b`'s geomeans (0.979–1.000) sit inside the same box as `base_a3`'s null reading
(0.984–1.011) for that round, and `after_c`'s (0.984–1.028) inside `base_a4`'s null (0.982–1.013).**
Both after legs are indistinguishable from a same-tree null on the default whole-frame cook
corpus — which is exactly what this release predicts: no program in the bench corpus carries a
`//!tex 0.25` pragma, so the default cook path is unchanged. Both null controls (opening and
closing) also confirm the box's established noise floor: CUDA configs ±1–2%, CPU-interpreter
configs noisier (rows to 4.33× on `cpu_on_warm` in the closing null alone — consistent with the
per-tree/compile-cache noise this box has shown before), which is why the geomean, not the row,
is the unit that carries a verdict.

**The four interactive-cost benches, read against their null neighbours:**

| bench / metric | `after_b` vs `a2` | null `a3` vs `a2` | `after_c` vs `a3` | null `a4` vs `a3` |
|---|---|---|---|---|
| roi_scrub — whole | 0.996 | 1.000 | 0.998 | 0.999 |
| roi_scrub — fixed / panning window | 0.999 / 1.000 | 1.002 / 1.004 | 0.989 / 0.989 | 0.996 / 0.995 |
| roi_scrub — pan+param | 0.996 | 1.001 | 0.984 | 1.002 |
| region_recook — CUDA whole / mid | 1.004 / 1.000 | 1.003 / 1.000 | 0.999 / 0.999 | 0.999 / 1.000 |
| param_scrub — recook median | 1.001 | 0.995 | 0.995 | 1.008 |
| cookqueue — queued_idle / preempt / under_load | 1.010 / 1.001 / 1.010 | 1.005 / 0.947 / 0.964 | 1.090 / 1.101 / 1.096 | 0.968 / 0.981 / 1.005 |
| `counts_cuda` — api/cuda rows moved | **0** | — | **0** | — |
| `counts_cuda` — frame rows moved (not gating) | 18 | — | 18 | — |

`after_b` reads **neutral** on every interactive metric — inside or tighter than its round's null
spread. `after_c`'s `roi_scrub` reads a consistent ~1–1.6% slower across all four of its rows
(0.984–0.998) against a null spread of 0.996–1.002 for the same round, and its `cookqueue`
reads ~9–10% *faster* on three rows (1.090–1.101) against a null spread of 0.947–1.005. Neither
reading is corroborated by `after_b`'s own pass over the same benches (both effects are absent
there), neither has a `counts_cuda` api/cuda row backing it (0 moved in both after legs — the
class of change this release makes does not touch the default per-cook call count), and
`cookqueue`'s established null band on this box is wide enough (down to 0.947 in this sitting's
own round-1 null) that a single-leg swing in either direction is the expected shape of the noise,
not a new phenomenon. **Read as suspicious, not confirmed**: worth a same-sequence re-leg if a
future sitting wants to settle it, but it does not meet the bar for an escalation, because the
counts — the actual gate — moved zero rows on the call/device axis in both after legs.

**`counts_cuda` frame rows** moved 18 in both `after_b` and `after_c` against their base
neighbours, consistently (same count, both rounds) — the expected signature of the stdlib split
(module boundaries move, `frames.total` and the per-module sum still conserve), not a per-cook
regression; the api/cuda axis is what gates, and it read 0/0.

## Verdict

**Neutral, as predicted.** The default whole-frame cook path shows no geomean or counts movement
outside what an identical tree produces against itself, on either after leg, across both rounds.
No escalation is opened. The `roi_scrub`/`cookqueue` wobble on `after_c` alone is flagged above so
a reader doesn't have to re-derive it, but it is not read as a regression or a speedup.

## Counts re-proof

`python -X utf8 -m pytest TEX_Wrangle\tests\test_bench2_counts.py -q -p no:cacheprovider`, run
from the `after` tree's parent directory on the reference box (CUDA present, so the CUDA-pinned
row executed rather than skipping):

```
.......                                                                  [100%]
7 passed, 1 warning in 5.05s
```

`rc=0`. The one warning is torch's own profiler notice about clearing events between cycles,
unrelated to the pins. (A `triton`-import stack trace and a `logging` teardown error also appear
in the raw `.out` file, both after the `7 passed` line and both artifacts of `torch.compile`
probing for an unavailable Triton backend during interpreter shutdown on this Windows box — they
do not affect the reported result and are the same class of harmless shutdown noise this box has
produced before.) All seven pins, including the CUDA-shaped row, matched on first run: the
device-independent counter rows this release's `after` tree produces on **sm_75** agree with the
values pinned against the development laptop's **sm_120**. No row differs between the two boxes;
no escalation.

## Reproducing it

Unchanged from `benchmarks/results/sitting_2026-09-20_sm75/README.md` §"Reproducing it":
`sitting.ps1` and `sitting_compare.py` reused as-is, per-leg cache directories emptied first, the
first leg of every freshly-materialised tree discarded, and read with
`python sitting_compare.py <results-dir> <reference-tag> <tag> [<tag> ...]`.
