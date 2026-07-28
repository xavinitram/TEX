# CACHE-8 — deep cache tiers: residency, packing, and the codec that isn't there

*Design note, v0.33.0. Companion: `tex_results.py` (the ladder), `tex_packing.py` (the
representations), `tests/test_v033_cache8.py`, `benchmarks/cache_capacity_bench.py`.*

## 0. The item, and what the measurement did to it

> **CACHE-8 (L, design doc).** Compressed cache tiers **and residency**: half/uint16 packing
> plus a fast lossless codec (LZ4-class) on RAM-evicted and disk-tier frames, chosen per tier
> by measured Pareto (compression ratio vs decode latency vs interactive feel — the Memory
> report's stated research goal, **run as a benchmark, not a belief**). CACHE-8 also owns the
> tier-1 question no item owned before: cooked frames may stay **VRAM-resident** while hot,
> with an access-frequency policy demoting them to RAM and promoting them back on reuse.
> Includes the honest research note on disk→GPU direct paths: investigate, measure, and record
> a go/no-go — not a commitment. *(doc 39 §3)*

The benchmark was run. It **rejected the codec** and **confirmed the residency tier**, and by
the standing rule ("anything here that contradicts a measured result when its design doc is
written loses to the measurement") this document ships the second and records the first as a
negative result rather than shipping a dial nobody should turn.

## 1. The Pareto (4K, `[1,2160,3840,4]` fp32 = 126.6 MB)

Decode is the column that decides it: encode is paid once, on eviction; **decode is paid on
every cache hit**. The two REFERENCE rows are what a codec must beat — not each other.

`benchmarks/cache_capacity_bench.py --resolution 4096`, RTX 2080 SUPER, torch 2.10:

| row | MB | ratio | encode ms | **decode ms** | maxerr |
|---|---|---|---|---|---|
| **fp16** | 128.0 | **2.00** | 10.5 | **12.9** | 2.4e-4 |
| **uint16 [0,1]** | 128.0 | **2.00** | 59.1 | **28.3** | 7.7e-6 |
| zlib-1 over fp32 | 166.6 | 1.54 | 6685 | 920 | 0 |
| zlib-6 over fp32 | 161.9 | 1.58 | 12699 | 868 | 0 |
| zlib-1 over fp16 | 46.8 | 5.47 | 1765 | 403 | 2.4e-4 |
| zlib-6 over fp16 | 38.6 | 6.63 | 3436 | 352 | 2.4e-4 |
| bz2 / lzma, any source | — | — | *skipped past a 32 MB cap: minutes per frame* | | |
| *REFERENCE — write the frame to disk / read it back* | 256.0 | 1.00 | **332** | **59.2** | 0 |
| *REFERENCE — move the frame VRAM→RAM / back* | 256.0 | 1.00 | **21.6** | **21.6** | 0 |

*Re-measured on a verified-quiet box. An earlier recording of this table was taken while other
work was running on the machine and a process kill landed mid-run; the numbers moved by under
2% and no verdict changed, but the tainted table is not the one shipped.*

The `bz2`/`lzma` rows are reported as skipped rather than dropped: "too slow to measure at 4K"
is the single most decision-relevant fact about them, and a missing row reads as
"not applicable".

**No LZ4.** `lz4`, `blosc`, `blosc2` and `zstandard` are all absent from the environment, and
a new third-party dependency is out of scope for a torch-only package (the same rule that bans
numpy). The stdlib's three codecs are what is actually available, and all three lose.

### Reading it

* **A general-purpose codec is not close.** zlib-1 over fp32 costs **6685 ms to encode** a frame
  that can simply be *written to disk* in 332 ms, and **920 ms to decode** one that can be *read
  back* in 59 ms. Compressing to keep a frame in RAM costs 15× more than not bothering and
  going to disk.
* **The break-even is a storage-bandwidth question, and it is answerable.** Reading 128 MB
  uncompressed at rate *R* costs `128/R`; reading the 47 MB zlib-1 version and decoding it costs
  `47/R + 0.403 s`. Compression wins only when **R < ~200 MB/s** — a network share or a spinning
  disk, never an SSD. And it would still cost 1765 ms of encode per evicted frame, on the drain
  path, which no storage medium makes acceptable.
* **Width is the whole win.** fp16 gets the full 2.00× for 12.9 ms of decode — 28× cheaper than
  the cheapest entropy coder that reaches a comparable ratio — and it applies to VRAM, RAM *and*
  disk simultaneously, because the frame is packed once, at admission.
* **uint16 is real but narrow.** 32× more accurate than fp16 inside `[0,1]` at identical size;
  clips outside it; **5.6×** the pack cost at 4K (59.1 ms against fp16's 10.5). Offered as `storage="uint16"`, never chosen
  automatically — sniffing a frame's range to pick a codec is exactly the silent auto-tuning
  S-5 forbids, and the first in-range HDR frame would be quietly clipped by the choice.

### Per-tier verdict (which is what "chosen per tier" was asking for)

| tier | representation | why |
|---|---|---|
| VRAM (tier 1) | as cooked, or fp16 at preview quality | decode is on the hot path |
| host RAM (tier 2) | whatever it was in VRAM — a demotion is a move, not a re-encode | bit-exactness |
| disk (tier 3) | whatever it was in RAM — already packed at admission | encode would cost 4× the write |
| *any tier* | **no entropy coding** | measured 1–2 orders of magnitude the wrong side of its own reference row |

## 2. Residency — the ladder that was missing

Before v0.33 the cache had **two** tiers and one rung between them: over budget, the LRU frame
was pickled to disk. For a CUDA frame that meant VRAM → disk → VRAM, and the middle of the
machine — host RAM, sitting empty — was not a place a frame could be.

```
v0.32     VRAM ────────────────────── disk
v0.33     VRAM ──── host RAM ──────── disk
```

Measured **through the shipped `ResultCache`**, not through a synthetic copy — so what the
table reports is the code that runs, lock discipline and drain path included
(`benchmarks/cache_capacity_bench.py`, 2048² frames, RTX 2080 SUPER):

| event | v0.32 path | v0.33 path | |
|---|---|---|---|
| a `put` that has to shed VRAM | spill to disk, **77.9–78.8 ms** | demote to host RAM, **5.7–5.9 ms** | **13.2–13.8×** |
| the `get` that wants it back | restore from disk, **117.6–118.4 ms** | promote from RAM, **6.3–8.6 ms** | **13.7–18.7×** |

Ranges, not single digits: these are two independent runs of the same rows, and the promote
number moved 6.3→8.6 ms between them. Quoting the best of the two as "20.1×" would be reading
a noise band as a measurement.

And the frame **stays a cache hit** rather than becoming a disk read. (The synthetic reference
rows in §1 are faster than the through-the-cache numbers because they measure a bare `copy_`
against a warm page cache; the table above is what a host actually pays.)

### Mechanics

* Entry slot 3 is **where the frame is**; new slot 6 is **where it belongs** (`home`, the cook
  device). They differ exactly while a frame is demoted, which is also how `get` recognises
  the reuse it should promote on.
* `_enforce_residency` runs under the lock and is **pure bookkeeping** — it queues victims.
  `_drain_demotes` performs the D2H *outside* the lock, per the module's standing rule that
  the lock covers structure and byte accounting and never full-frame copies.
* Unlike a spill, **a demotion never removes the entry**. It stays in `_ram`, on CUDA,
  answering `get` with the correct pixels until the host copy exists; only then is slot 0
  swapped under the lock. There is no window in which the frame is missing, so there is no
  window anyone has to reason about.
* `put` runs residency **before** the total budget: relieving VRAM the cheap way first means a
  frame only reaches the disk when host RAM is full too.
* `evict_bytes` — the CACHE-5 governor hook — **demotes instead of evicting** when the governor
  asks for CUDA bytes and the tier is armed. The governor gets exactly the resource it asked
  for, and the cache keeps its contents.
* A demoted frame that later spills writes its **home** device, not its current one. Writing
  slot 3 there would have turned the ladder into a one-way trip to the CPU, visible only under
  memory pressure. `test_v033_cache8_a_spilled_demoted_frame_comes_back_to_its_home` is that
  bug's tripwire.

### The policy, stated so it can be argued with

Victims are chosen by **LRU**; a demoted frame is promoted on its **next hit**. The report asks
for an access-*frequency* policy, and this is a recency one. The justification is that a frame
cache's access pattern is a playhead — a scrub touches near-frames both most recently and most
often, so the orderings largely coincide — and the honest part is that this is an argument, not
a measurement. `stats()` therefore reports `demotions`, `promotions` and `demoted` precisely so
the question can be reopened with data. A frequency-weighted victim choice is a change to
`_enforce_residency` alone.

### Off by default

`_vram_budget` starts `None` and GOV-1's `balanced` preset carries `vram_mb: None`. A cache
that was never given a ceiling behaves exactly as v0.32 — it does not start moving frames
between devices because the package was upgraded. `performance` sets 2048 MB, `efficient`
256 MB (the tightest, because that profile is chosen precisely when something else wants
the GPU).

## 3. GOV-1's fourth knob

The v0.32 item text reserved "compression aggressiveness (from v0.33)". There is nothing to be
aggressive about, so the slot carries **`vram_mb`** instead — the knob the measurement says buys
capacity. `_armed_caches` changed from `cache -> the one remembered budget` to
`cache -> {knob: remembered default}`, and knob→setter binding moved into a `_CACHE_KNOBS`
table: without that, `balanced` could restore the frame budget and would silently leave the
residency ceiling wherever `efficient` put it — the identical bug GOV-1 already shipped once,
one knob over.

## 4. Research note: disk→GPU direct paths — **NO-GO**

GPUDirect Storage (cuFile) lets an NVMe drive DMA straight into VRAM, skipping the host bounce.
Findings:

1. **No API surface.** torch 2.10 exposes no cuFile binding. Reaching it means ctypes against
   `cufile.so`/`cufile.dll` plus an aligned-buffer registration protocol — a platform-specific
   dependency in a package whose portability floor is "torch and the stdlib".
2. **Linux-only in practice.** GDS is not supported on Windows, which is this project's
   primary development and benchmark platform.
3. **It is not the bottleneck.** The path it would replace is *disk read + H2D* = 35.7 + 11.1 =
   46.8 ms at 4K. GDS removes at most the H2D half. Meanwhile the residency tier removes the
   **disk hop entirely** for the frames that matter, at 11.1 ms — so the optimisation with the
   large, portable, already-measured win is the one this item shipped.
4. **Revisit condition, stated so it is checkable:** torch exposing a first-party GDS API, *and*
   a workload whose working set genuinely exceeds host RAM (8K plates, long playback ranges).
   Neither holds today.

## 5. What this item did not do

* **No compression.** Measured, rejected, recorded above. `test_v033_cache8_no_compression_
  path_is_switched_on` greps `tex_results.py` for codec names so a future "small" addition has
  to argue with the measurement rather than slip past it.
* **The demote D2H is synchronous.** It blocks the draining thread for 10.8 ms at 4K. Making it
  asynchronous is exactly **XPU-2**, this release's next item — the drain path is where its
  frame handle plugs in, and doc 39's dependency spine already reads `XPU-2 → CACHE-8 spill
  path`. Shipping the ladder synchronously first means the correctness is pinned by tests
  before the concurrency is added under it.
* **No VRAM→VRAM tiering across multiple GPUs.** `home` is a device string and would support
  it; nothing else does, and no measurement asked for it.
