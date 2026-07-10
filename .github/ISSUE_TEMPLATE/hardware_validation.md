---
name: Hardware validation report
about: Share whether TEX's perf gates hold on your GPU (Ampere / Ada / Blackwell wanted!)
title: "[HW-VALIDATION] <your GPU> (sm_XX)"
labels: hardware-validation
---

TEX's performance gates (the CUDA-graph crossover, the fp16 threshold, the autotier
thresholds) were **calibrated on a single GPU** — an RTX 2080 SUPER (Turing, sm_75). We
want to know whether those constants hold on other architectures. If they don't, your
report drives a per-architecture recalibration.

## How to generate the report

From the TEX Wrangle directory, with your ComfyUI Python environment:

```
python -m TEX_Wrangle.tex_cli validate-hw
```

It runs in under a minute, needs no arguments, and writes
`benchmarks/results/validate_hw.md` (+ `.json`). It SKIPs any lane it can't run
(no Triton, TF32 on pre-Ampere, CPU-only) with a clear reason — it will not fail.

## Paste the report here

<!-- paste the full contents of benchmarks/results/validate_hw.md below -->

```
(paste validate_hw.md here)
```

## Notes (optional)

- Anything unusual you noticed?
- Driver / OS / torch build if not already in the report.
