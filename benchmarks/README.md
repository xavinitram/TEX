# TEX Benchmark Suite

Reproducible performance benchmarks for the TEX Wrangle ComfyUI node.

## Quick Start

```bash
cd TEX_Wrangle

# Standard run: 256/512/1024, batch=1, warm cache (~10 min)
python benchmarks/run_benchmarks.py

# Quick sanity check: 512 only (~2 min)
python benchmarks/run_benchmarks.py --quick

# Full matrix: 256-4096, batch 1+4, warm+cold (~60+ min)
python benchmarks/run_benchmarks.py --full --cold

# Compare against a baseline
python benchmarks/run_benchmarks.py --compare benchmarks/results/v0.10.0.json
```

## Structural counts, not times: `host_path_counts.py`

Everything else in this directory measures **milliseconds**. `host_path_counts.py` measures
**integers**: how many times each seam on the cook path is entered per interactive tick, for
the paths an embedding host drives. It exists because wall-clock on a development laptop cannot
gate a regression — `docs/roadmap.md` §10 item 3 records a byte-identical tree tripping the 0.95
stop-ship threshold against itself — while the per-tick counts are exact and repeat run after
run. `tests/test_bench2_counts.py` pins them, so an interactive-path regression reds in CI on a
machine with no GPU.

```bash
python benchmarks/host_path_counts.py                         # both devices if present
python benchmarks/host_path_counts.py --device cpu --res 96 --window 48 --ticks 4
python benchmarks/host_path_counts.py --save results/counts_head.json
python benchmarks/host_path_counts.py --compare results/counts_head.json   # rc 1 on drift
python benchmarks/host_path_counts.py --selftest              # prove the spies are not inert
```

It drives `examples/host_demo.py::RoiComp` through seven scenarios (`prewarm`, `source_edit`,
`terminal`, `midgraph`, `pan`, `all_dirty`, `lint`) plus an eighth, `node_scrub`, which drives
the ComfyUI node instead of the comp — two `check_lazy_status` rounds and the node's own
`execute` per tick, because the lazy tier hangs off a `forgive_dead_refs` flag only that node
passes, and the other seven therefore cannot see it. It reports, per tick, API call counts, TEX
Python frames per `module:function`, and — on CUDA — kernel launches, memcpys and allocator
statistics. Each row carries a `stable` flag (`min == max` over the steady ticks); only stable
rows are compared, because a row that disagrees with itself cannot gate anything.

The design note, the measured signature at head and the deferred timing tier are in
`docs/host-path-counts.md`.

## What It Measures

Each TEX program is measured with adaptive run counts (10-50 runs, stopping
early when coefficient of variation drops below 5%):

| Metric          | Description                                      |
|-----------------|--------------------------------------------------|
| **compile_ms**  | Median time for lex + parse + typecheck + optimize |
| **interp_ms**   | Median interpreter execution time                |
| **total_ms**    | Median end-to-end time                           |
| **cv_percent**  | Coefficient of variation (measurement stability) |
| **p5_ms / p95_ms** | 5th and 95th percentile of total time         |
| **mem_mb**      | Peak memory delta during execution               |

## Programs

### Synthetic Micro-Benchmarks (15)

Isolate specific engine capabilities:

| Name             | Tests                                  |
|------------------|----------------------------------------|
| passthrough      | Dispatch overhead (identity copy)      |
| math_chain       | Trig functions + arithmetic            |
| function_heavy   | 10 chained stdlib calls                |
| branch_simple    | Single if/else                         |
| branch_nested    | Nested if/else (4 branches)            |
| for_10           | Short loop (10 iterations)             |
| for_100          | Medium loop (100 iterations)           |
| while_loop       | While loop with convergence            |
| vector_ops       | vec3/vec4 construction + swizzle       |
| array_ops        | sort, sum, min, max                    |
| string_ops       | String functions (no spatial context)  |
| noise_perlin     | Single perlin noise call               |
| noise_fbm        | fBm noise (6 octaves)                  |
| sample_bilinear  | Bilinear image sampling                |
| sample_lanczos   | Lanczos image sampling                 |

### Real-World Examples (36)

All programs from `examples/` — blur, edge detection, vignette, color
grading, noise generation, sampling, etc.

## Flags

```
--full              Full matrix (all resolutions, batches, cache modes)
--quick             Quick: 512x512 B=1 warm only
--resolution N      Single resolution (default: 256, 512, 1024)
--batch N           Single batch size (default: 1)
--cold              Include cold-cache (recompile every iteration)
--device cpu|cuda   Force device (default: cpu)
--precision P       fp32 (default), fp16, bf16
--save PATH         Save results to specific path
--compare PATH      Compare current run vs saved baseline
--examples-only     Skip synthetic programs
--synthetic-only    Skip example programs
```

## Output Format

Results are saved as JSON in `benchmarks/results/`:

```json
{
  "system": {
    "timestamp": "2026-03-20T14:30:00",
    "platform": "Windows",
    "python": "3.12.11",
    "torch": "2.8.0+cu129",
    "cpu_name": "...",
    "cpu_cores": 8,
    "gpu_name": "...",
    "tex_version": "0.11.0",
    "git_commit": "ffaff43",
    "git_dirty": true
  },
  "config": {
    "min_runs": 10,
    "max_runs": 50,
    "warmup": 5,
    "target_cv": 0.05
  },
  "results": [
    {
      "program": "passthrough",
      "resolution": 512,
      "batch": 1,
      "cache_mode": "warm",
      "compile_ms": 0.0,
      "interp_ms": 1.234,
      "total_ms": 1.234,
      "cv_percent": 3.2,
      "p5_ms": 1.100,
      "p95_ms": 1.400,
      "mem_mb": 0.0,
      "num_runs": 15
    }
  ]
}
```

## Running on Another System

1. Install TEX Wrangle as a ComfyUI custom node
2. Activate the ComfyUI Python environment
3. Run from the TEX_Wrangle directory:

```bash
python benchmarks/run_benchmarks.py --save benchmarks/results/my_system.json
```

4. To compare against a reference:

```bash
python benchmarks/run_benchmarks.py --compare benchmarks/results/v0.10.0.json
```

## Interpreting Comparison Output

```
Program                    Res       Baseline    Current   Speedup  Verdict
blur                      512x512    45.000ms   30.000ms    1.50x  FASTER
grayscale                 512x512     3.000ms    2.900ms    1.03x  ~same
noise_fbm                 512x512   200.000ms  220.000ms    0.91x  SLOWER

Geometric mean speedup : 1.250x
Faster (>1.05x)        : 30/45
Slower (<0.95x)        : 2/45
Neutral                : 13/45
```

- **Speedup > 1.05x** = meaningful improvement
- **Speedup < 0.95x** = meaningful regression
- **Geometric mean** = the single-number summary across all benchmarks
