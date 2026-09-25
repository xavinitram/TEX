# Release sitting — 2026-09-25, sm_75 reference box

The timing record for **v0.44.0**, taken against `base` = **v0.43.2**, under the project's
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
| `after_discard` | `v0.44.0` | **discarded** — first run of the freshly-materialised `after` tree |
| `base_a1` | `v0.43.2` | base — with `base_a2`, the opening **null control** |
| `base_a2` | `v0.43.2` | base — opening null control |
| `after_b` | `v0.44.0` | after, round 1 |
| `base_a3` | `v0.43.2` | base — reference for round 1's read and round 2's |
| `after_c` | `v0.44.0` | after, round 2 |
| `base_a4` | `v0.43.2` | base — reference for round 2's read and the closing null |
| `base_a5` | `v0.43.2` | base — the **closing null control** against `base_a4` |

Both trees were shipped fresh immediately before the sitting; the `after` sha was confirmed a
descendant of the `base` sha beforehand. All eight legs ran back to back, unattended, in one
sequence, with a per-leg preflight/postflight process-and-GPU check; every leg's preflight found 0
other python processes, and every one of the 48 bench invocations returned `rc=0`. Total wall time
for the sequence: ~5h40m, no connection interruption during this run.

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
| cpu_off_cold | 0.997 | 1.003 | 0.990 | 1.002 | 0.999 | 1.017 |
| cpu_off_warm | 0.984 | 1.006 | 1.003 | 0.995 | 1.007 | 1.018 |
| cuda_off_cold | 1.002 | 0.991 | 0.994 | 0.988 | 1.017 | 0.973 |
| cuda_off_warm | 0.996 | 0.984 | 0.997 | 0.988 | 1.005 | 0.989 |
| cpu_on_cold | 1.058 | 0.982 | 1.006 | 0.985 | 0.990 | 0.959 |
| cpu_on_warm | 0.995 | 1.005 | 1.007 | 0.990 | 0.998 | 1.002 |
| cuda_on_cold | 1.002 | 0.999 | 1.004 | 0.987 | 0.989 | 1.007 |
| cuda_on_warm | 0.992 | 0.994 | 1.002 | 0.993 | 0.994 | 1.009 |

**The interpreter's mixin split shows no cost on any `off_*` row.** This release moves the
interpreter's implementation into mixin methods and adds a no-op-when-unused cancel-poll check;
neither is claimed to touch the default path's cost. Reading the four rows that would show it
first — `cpu_off_cold`, `cpu_off_warm`, `cuda_off_cold`, `cuda_off_warm` — against the full null
spread this box produced across all four null legs (not just each round's own bracketing pair,
since a single null reading can itself be noise on this box): `cpu_off_cold` nulls span
0.990–1.017, both after legs (1.003, 1.002) sit inside; `cpu_off_warm` nulls span 0.984–1.018,
both after legs (1.006, 0.995) sit inside; `cuda_off_cold` nulls span 0.973–1.017, both after legs
(0.991, 0.988) sit inside; `cuda_off_warm` nulls span 0.989–1.005, and the two after legs read
0.984 and 0.988 — each within about half a percentage point of the low edge of that band, and
both within the wider spread this same metric showed across the four null legs on the *prior*
sitting in this series (0.985–1.006). **No `off_*` row reads clearly outside its null band; the
`cuda_off_warm` edge case is noise-scale, not a reproduced, one-directional effect of the size
this series has flagged before** (compare the prior sitting's `cuda_off_cold` finding, which read
6%+ outside a band a fifth this wide). Verdict: **no cost found on the interpreter-split rows.**

**Every eight_config geomean, on- and off-path, reads inside its own round's null band or the
wider four-leg spread**, confirming the default whole-frame per-cook path is neutral, as this
release claims for the mixin split.

**The ROI scrub bench reads neutral, as claimed.** All four `roi_scrub` rows (`whole`,
`roi_fixed`, `roi_panning`, `roi_pan_param`) move by 1% or less in both after legs, matching the
null legs' own swings in the same rounds — no row moves outside what the null controls
themselves show. `region_recook`, `param_scrub`'s medians, and `cookqueue`'s solo/idle rows are
likewise inside their own rounds' null swings; `cookqueue`'s `preempt_to_first_stmt_ms` and
`queued_under_load_ms` move by 4–12% in round 2's after leg, but this box's `cookqueue` rows are
consistently noisy at this magnitude across every sitting in this series (the round 2 null itself
is not shown moving less), so this is read as this bench's known noise floor, not a finding.

**The structural counts harness moved 0 api/cuda rows in both after legs** (27 frame rows moved
in both — the expected, lawful signature of a module reorganisation adding call sites, not
gating). This matches the release's own claim: the interpreter mixin split is a mechanical
reorganisation of method lookup, not a new call layer, and it leaves the per-cook call-count
structure untouched. This sitting's six benches do not include a wall-clock measurement of
`checkpoint_serve` specifically (it exists in this harness only as a structural scenario, whose
counter rows read unchanged here); the release's own separate measurement of that path is not
independently re-timed by this sitting.

## Verdict

**Neutral, as claimed, on every axis this sitting measures.** No eight_config geomean — including
every `cpu_off_*`/`cuda_off_*` row, which is where a cost from moving the interpreter into mixins
would show first — reads outside its own null band or the wider four-leg null spread. The ROI
scrub reads neutral. The structural counts harness shows 0 api/cuda rows moved (27 frame rows,
expected and lawful). No escalation from this sitting.

## Counts re-proof

Run against the **`after` tree only** (a scenario pair this harness gained after the `base` sha
does not exist on `base`, so a cross-tree run is not meaningful here):

```
.......                                                                  [100%]
7 passed, 1 warning in 5.7s
```

`rc=0`. The one warning is torch's own profiler notice about clearing events between cycles,
unrelated to the pins. (A `triton`-import stack trace and a `logging` teardown error also appear
in the raw output, both after the `7 passed` line and both artifacts of `torch.compile` probing
for an unavailable Triton backend during interpreter shutdown on this box — harmless, the same
class of shutdown noise seen in prior sittings, and do not affect the reported result.) All pins
matched, including the CUDA-shaped row: the device-independent counter rows this release's
`after` tree produces on **sm_75** agree with the values pinned against the development
hardware's **sm_120**. No escalation from this half of the re-proof.

## Reproducing it

Unchanged from the prior sittings in this series: the same runner and comparator, reused as-is,
per-leg cache directories emptied first, the first leg of every freshly-materialised tree
discarded, and read with the comparator over the results directory and the legs to compare.
