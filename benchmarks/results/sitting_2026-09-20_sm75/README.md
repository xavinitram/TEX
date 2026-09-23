# Release sitting — 2026-09-20, sm_75 reference box

The timing record for **v0.38.0 "Count, don't time"**. It exists because counts, not times, are
this project's CI gate (`docs/host-path-counts.md` §1): wall-clock on the development hardware
cannot decide anything by itself — a byte-identical tree has tripped the 0.95 stop-ship
threshold against itself, and single rows span 0.70× to 2.32×. Timing is therefore measured
**once per release, at a sitting, on a quiet box**, under the rules in `docs/roadmap.md` §10
item 3. This directory is that sitting: every comparison read off it, and
the commands that reproduce it.

`.comfyignore` excludes `benchmarks/`, so nothing here reaches the published registry archive.

**The raw per-leg result files are no longer in the repository.** The sixty `<tag>_<bench>.json`
files this sitting produced (about 106,000 lines of pretty-printed samples) were removed from the
repository's history on 2026-09-23, under the rule in `docs/roadmap.md` §10 item 3 that a sitting
commits its reading, not its raw legs. The maintainer keeps them outside the repository. Every
read and verdict below was taken from them and is reproduced here in full, and the `read_*.txt`
files are the comparator's own output.

## The box

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 2080 SUPER (**sm_75**) — the reference box, a desktop |
| torch / python | 2.12.0+cu130 / 3.11.9 |
| state | quiet: GPU utilisation and memory checked either side of every leg; no other Python process |

This is deliberately **not** the sm_120 development laptop, which is shared with other work and
whose clocks idle down. The device-independent counter rows in `tests/test_bench2_counts.py`
read identically on both machines; that agreement is what makes them gateable, and it is also
what lets a timing sitting run on a box the laptop's own benchmarks were never taken on.

## The legs

Ten legs. Every leg ran the same six benches (five timing benches plus the counts harness) in
the same order, each with **its own cache directory, emptied before the leg started**.

| leg | tree | role |
|---|---|---|
| `base_a1` | `v0.37.0` (`dfe7c38`) | base |
| `base_a2` | `v0.37.0` | base — with `base_a1`, the opening **null control** |
| `base_a3` | `v0.37.0` | base |
| `base_a4` | `v0.37.0` | base — the reference leg of the closing read |
| `base_a5` | `v0.37.0` | base — the **closing null control** |
| `after1_b` | PERF-1 + PERF-2 (`c9dde51`) | after, round 1 |
| `after2_c` | PERF-1..5 (`4ee435d`) | after, round 2 |
| `after2_d` | the same tree as `after2_c` | after, round 2 **repeated** |
| `after3_discard` | the round-3 tree (`8f38f82`) | **discarded** — first run of a newly materialised tree |
| `after3_e` | the same tree, second run | after, round 3 — **the reported leg** |

The round-3 tree is `8f38f82`: PERF-1..8, BENCH-2 and BENCH-3, NEG-1/2/3. The legs do not
record their own sha, because they predate the `--save` provenance fields (measured sha, cache
directory, cache warmth) that SIMP-1 added to the counts harness later the same day — which is
precisely why those fields now exist.

## The comparisons

Each `read_*.txt` is the output of `sitting_compare.py` over the legs named in its first line.
Ratios are `reference / leg`, so **>1 means the leg is faster**.

| file | legs | what it answers |
|---|---|---|
| `read_null_a1_a2.txt` | `base_a1`, `base_a2` | the **null control**: what does the same tree read against itself? |
| `read_sitting1_a2_b_a3.txt` | `base_a2`, `after1_b`, `base_a3` | round 1, base → after → base |
| `read_sitting2_a3_c_a4.txt` | `base_a3`, `after2_c`, `base_a4` | round 2, base → after → base |
| `read_repeat_a4_d.txt` | `base_a4`, `after2_d` | round 2 repeated against a fresh base leg |
| `read_sitting3_a4_e.txt` | `base_a4`, `after3_e` | round 3, two legs — superseded by the next row |
| **`read_sitting3_a4_e_a5.txt`** | `base_a4`, `after3_e`, `base_a5` | round 3, **base → after → base — the release's reported reading** |

Quote the last one. It is the only read in this directory whose null control ran in the same
sequence as the claim it protects, which is the difference between a number and a measurement.

`sitting_breakdown.txt` is the per-program median table behind the eight-config geomeans, for
the one configuration (`cpu_on_cold`, compiled status, n=56) that a reported regression was
opened on. It is included because a geomean without its rows cannot be argued with.

## The null band, and how to read the result

**Per-config geomeans are the interpretable unit; individual rows are not.** The closing read's
three legs, all ratios against `base_a4`, with the null leg in the third column:

| measurement | `after3_e` | `base_a5` (identical code) |
|---|---|---|
| eight-config corpus, per-config geomean, all eight | 0.996 – 1.027 | **1.003 – 1.047** |
| `roi_scrub` — panning window **plus** a moving parameter | **1.215** | 1.013 |
| `roi_scrub` — fixed window / panning window | 1.041 / 1.040 | 1.000 / 1.001 |
| `roi_scrub` — whole frame | 1.014 | 0.998 |
| `region_recook` — CUDA 2048², mid region / whole frame | 1.028 / 1.008 | 0.999 / 1.002 |
| `param_scrub` — recook median | 1.030 | 1.009 |
| `param_scrub` — scrub / static median | 1.022 / 1.023 | 1.005 / 1.008 |
| `param_scrub` — worst tick after warm-up | 1.334 | 0.988 |
| `counts_cuda` — structural rows moved | **23**, every one predicted | **0** |

**On the eight-config corpus the null leg's spread is wider than the claim's.** Identical code
returned 1.003 – 1.047; the release returned 0.996 – 1.027. So the corpus — the default
whole-frame cook path — is **neutral**, which is what `AGENTS.md` invariant 7 requires, and the
release claims no speedup there. The opening null pairing (`read_null_a1_a2.txt`) says the same
thing from the other end of the sitting: 0.967 – 1.033, with individual rows from 0.40 to 2.49.

**On the four benches that isolate an interactive cost, the null leg sits at 0.99 – 1.01 and
the release does not.** That gap is the result, and every row of it was predicted by a counter
before it was timed. The counts column is the cheapest half of the same argument: a null leg
owes a null reading to the counters too, and this one gave 0 against the release's 23.

## Two artefacts this sitting measured, and the protocol they bought

**First run per tree.** `read_sitting1_a2_b_a3.txt` shows `region_recook cpu/n50/2048/whole_all`
at **1996.74 ms** on `base_a3` against ~900 ms on every other leg of the same sitting, including
the same tree immediately afterwards (`base_a4`, 909.13 ms). A 2.2× phantom, on one row, from
one tree's first run. **Therefore: the first leg run against any newly materialised tree is a
discard leg** — run it, throw it away, and report from the second. `after3_discard` is that leg,
kept here so the protocol is visible rather than described.

**A warm shared cache is not a comparison.** Each leg gets its own cache directory, emptied
before the leg starts, because two legs sharing one produce rows that look structural and are a
second leg reading what the first wrote. Both rules are now standing law in
`docs/brief-conventions.md` §"Two measurement rules that are not negotiable".

## Reproducing it

Per leg, from the directory that **contains** the checkout being measured (never from inside a
directory an embedding host scans — a host will import the tree you are trying to measure), with
a cache directory of this leg's own that starts empty:

```
python benchmarks/eight_config_bench.py  --resolution 512 --save  <tag>_eight_config.json
python benchmarks/roi_scrub_bench.py     --device cuda --resolution 1024 --roi 512 --save <tag>_roi_scrub.json
python benchmarks/param_scrub_bench.py   --device cuda --res 512 --save <tag>_param_scrub.json
python benchmarks/region_recook_bench.py --roi 512 --save <tag>_region_recook.json
python benchmarks/cookqueue_bench.py     --device cuda --res 512 --save <tag>_cookqueue.json
python benchmarks/host_path_counts.py    --device cuda --res 1024 --window 512 --ticks 8 \
                                         --prof1 off --save <tag>_counts_cuda.json
```

Then read the legs against each other:

```
python sitting_compare.py <results-dir> <reference-tag> <tag> [<tag> ...]
```

The counts leg is the one that gates. Its CI-shaped form is device-independent and cheap —

```
python benchmarks/host_path_counts.py --device cpu --res 96 --window 48 --ticks 4 --prof1 off
```

— and it is what `tests/test_bench2_counts.py` pins and `tools/gate.py --tier full
--counts-baseline <json>` runs. A counts comparison's **verdict counts the call and device rows
only**; the `frames.*` census prints under its own heading with its totals, because it moves for
every lawful change that adds a call or splits a module.
