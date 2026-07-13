# XPU transfer scheduling — design note (v0.20 shipped / v0.21 roadmap)

Mixed-device chains (a CPU-cooked TEX node feeding a CUDA-cooked one, or vice
versa) are the "XPU workflow" case. This note records what v0.20 shipped, the
measured numbers behind it, and the two follow-ups deliberately NOT built yet.

## Copy count vs copy scheduling

Auto-device chains were already optimal on copy COUNT before v0.20: outputs stay
on the compute device (`prepare_output` never forces `.cpu()`), the next node's
`auto` follows the incoming tensor's device, and the same-device guard skips the
`.to()` entirely. A forced CPU→CUDA handoff is exactly ONE direct H2D hop — no
round-trips. What v0.19.1 lacked was copy SCHEDULING: the one hop was a blocking
`t.to(device)` from pageable memory, executed before the first kernel launches.
It could not be async even in principle — CUDA only DMAs asynchronously from
page-locked (pinned) memory, and PyTorch silently downgrades `non_blocking=True`
to a sync copy from pageable sources.

## v0.20 — pinned egress + non-blocking ingestion (shipped)

- **Pinned egress** (`tex_marshalling._pinned_clamp01` / `_pinned_contiguous`):
  a CPU cook with CUDA present writes IMAGE/MASK/LATENT outputs ≥ `_PIN_MIN_BYTES`
  (1MB) into page-locked memory. Same write either way; torch's caching host
  allocator amortizes the page-lock cost across cooks. `unwrap_latent`'s
  BCHW→BHWC materialization pins too, so latent chains keep pinned-ness through
  to the next ingestion.
- **Non-blocking ingestion** (`to_fp32_if_int_image(device=...)`, the single
  ingestion source for interpreter AND codegen): a pinned CPU binding headed to
  CUDA issues `.to("cuda", non_blocking=True)` — the DMA rides in the background
  while Python continues (remaining bindings, coordinate builtins, dispatch).
  Stream ordering makes the first consuming kernel wait exactly until the copy
  lands: no events, no manual sync, bit-identical (the DataLoader pattern).
  Pure device moves only — dtype-converting copies stay synchronous; D2H is
  never non_blocking (a CPU-side read could observe an in-flight buffer).

Measured (RTX 5070 Ti Laptop, PCIe, interpreter cook wall, mixed CPU→CUDA):
1024² 16MB — 1.55→1.41ms (**1.10x**); 2048² 64MB — 11.94→8.46ms (**1.41x**).
The saving is bounded by the Python-side work available to hide the copy behind
(~0.5–1.5ms warm; more on cache-miss cooks where parse/typecheck/codegen add
milliseconds). All-CPU and all-GPU chains are untouched.

Guards: `_pin_worthwhile` requires CPU tensor + ≥1MB + `torch.cuda.is_available()`;
every pinned alloc has a pageable fallback (host-memory pressure can fail
`cudaHostAlloc`); pinned memory reads/writes like normal CPU memory for any
downstream CPU consumer. Tests: XPU-1..4 in `tests/test_v020_phase1.py`.

## v0.21 candidate — 3-stream tiled pipeline (NOT built)

Full copy/compute overlap — upload tile i+1 while computing tile i and
downloading tile i-1 on three streams (H2D ∥ compute ∥ D2H). The two
prerequisites already exist: `is_tile_safe` (pointwise-ness gate, single-sourced
in tex_memory) and `run_tiled` (M-4 strip machinery). What's missing is the
stream choreography + per-strip pinned staging ring. Only pays when transfer
time ≈ compute time (big frames, simple programs); gate it on a measured
crossover like every other tier. Do NOT build it before measuring how often
that regime occurs in real chains.

## Speculative (design-only, do not build without a driving workflow) —
## return-time upload

A CPU TEX node could START the async upload of its (pinned) output at return
time when ComfyUI graph introspection shows the downstream consumer will cook
on CUDA — the copy would ride ComfyUI's inter-node bookkeeping gap instead of
the next node's ingestion. Requires peeking at the workflow graph (fragile
across ComfyUI API versions) and a handoff convention for "tensor with copy in
flight" (an event or the CUDA-tensor future). Recorded so it isn't re-derived;
not scheduled.
