# XPU-2 — engine-owned async D2H egress

*Design note, v0.33.0. Companion: `tex_runtime/streams.py`, `tests/test_v033_xpu2.py`.*

## 0. The rule this lifts, and why it was right

Since v0.20 the tree has carried one line, in `tex_marshalling.to_fp32_if_int_image`:

> D2H is never non_blocking (a CPU-side read could observe an in-flight buffer).

It is not a limitation anyone chose. It follows from the type: a `torch.Tensor` has nowhere to
keep *"not yet"*. Hand one to a consumer and the consumer reads it; there is no third option.

XPU-2 does not weaken the rule. It changes the **object**:

> A D2H may be non_blocking when its result is carried by a `FrameHandle`, because a handle
> has nowhere to be read from that does not fence first.

## 1. Why this is admissible now and was not before

The register shelved "return-time transfer" twice, on two objections, and recorded that both
are properties of the *host* rather than of the idea:

| objection | under ComfyUI | under engine custody |
|---|---|---|
| API fragility | a cook's output goes to a third-party node graph that has never heard of a handle; the first `.cpu()` on the wrong side is a silent wrong frame | every consumer is in this repository |
| custody loss | the engine cannot know when the consumer is done reading | the engine *is* the consumer |

So the safety argument is a **finite, reviewable consumer list**, and it is asserted as a fact
rather than a convention: `test_v033_xpu2_is_engine_only` greps the package and requires that
the only importer of `tex_runtime.streams` is `tex_results.py`, and that `tex_node.py` cannot
reach a handle at all. If a third module ever imports it, that test goes red and the custody
argument gets re-made deliberately instead of eroding.

## 2. The handle

```python
h = egress(cuda_tensor)      # returns immediately; the DMA is in flight
h.shape, h.dtype, h.nbytes() # safe at any time — decided when the copy was ISSUED
h.is_ready()                 # never blocks
h.tensor()                   # THE FENCE. the only supported route to the bytes
h.unsafe_buffer()            # no fence. two callers only; the name is the documentation
```

Three decisions worth stating:

**Metadata never fences.** How big a frame is was decided when the copy was issued, not when it
lands, so the byte accounting a cache does on every eviction never waits for a DMA. That
asymmetry is the reason this is a handle and not a `(tensor, event)` tuple — the useful part of
async egress is exactly the work a consumer can do while the copy is in flight.

**A handle pins its SOURCE until the fence.** Dropping the last reference to a tensor a DMA is
still reading returns its block to torch's caching allocator, which may hand it to another
stream with no ordering relationship to ours. The reference is released the moment `wait()`
returns, so a handle held for a while does not keep a VRAM frame alive.

**`egress` declines rather than faking.** A CPU source, no CUDA, a dtype conversion (torch keeps
those synchronous), a size outside the pinned band, or `retained=True` all return an
*already-complete* handle — same interface, no event. The caller writes one code path. An API
whose fast path has a different shape from its slow path grows a consumer that only fences on
one of them.

**Observing completion releases.** `is_ready()` returning True drops the source and the event,
exactly as `wait()` does. Without that a poll-based consumer that never fences would keep both
the source's VRAM block and the page-locked host buffer alive indefinitely — measured at 64 MB
still allocated at 2048² after the source name was deleted.

## 3. `retained=True`, and the correction that produced it

CUDA can only DMA asynchronously to **page-locked** host memory; torch silently downgrades
`non_blocking=True` from pageable sources. So async egress requires a pinned destination.

The first version of the residency demote used one and then cloned to pageable, to release the
lock — a second full pass over the frame, to buy an asynchrony worth less than the pass. The
distinction that resolves it is not performance, it is **lifetime**:

> Pinning is for **transient staging**. A destination the caller will **retain** must not be
> page-locked, because pinned pages are unswappable and torch's caching host allocator holds
> freed blocks for the process lifetime — a retained pinned frame is a slow leak of memory the
> OS can never reclaim. That is the same reasoning behind the shipped 256 MB `_PIN_MAX_BYTES`.

The parameter is named for the fact the **caller** has (*I am keeping this buffer*), not for what
egress allocates. It was `staging=` first, and that name required the caller to already know
that pinning is what enables asynchrony — which is not a fact a caller should need. Everything
else derives from `retained`.

| consumer | mode | why |
|---|---|---|
| `_spill` — pickle a frame to disk | async | the buffer lives for one `pickle.dump`; two `getsize` syscalls overlap the DMA |
| `_drain_demotes` — VRAM → host RAM | `retained=True` | the buffer becomes the cache entry and is held for as long as the frame is demoted |

## 4. Honest accounting of the win — measured, and smaller than the machinery

Splitting the two effects (pinned destination vs the asynchrony on top of it), 512² / 1024² /
2048² / 4096²:

| | 512² | 1024² | 2048² | 4096² |
|---|---|---|---|---|
| pinned vs pageable blocking | **1.22×** | 1.09× | 1.14× | **0.93×** |
| async vs pinned blocking | 1.06× | 1.02× | 1.01× | 1.05× |

Two things fall out, and both are worth stating plainly rather than rounding away:

* **The asynchrony itself is noise.** D2H copies issued on one stream serialise — one DMA
  engine per direction — so a batch does not overlap with itself. What can overlap is the copy
  against *CPU* work, and in `_spill` that CPU work is two filesystem `stat` calls. The
  per-call cost of the event machinery is ~6.3 µs (`current_stream` 4.24, `Event()` 0.76,
  `is_available` 1.12), which is the entire budget it has to win back.
* **At the top of the pinned band the pinned destination is a net LOSS** (0.93× at 4096²/256 MB).
  The `_PIN_MAX_BYTES` cap exists for an unswappable-memory reason, and this says the same cap
  is roughly where the *latency* argument runs out too.

So what XPU-2 delivers in v0.33 is **the contract, exercised** — not a speedup:

* the object v0.34's async-write item is specified in terms of ("a completed frame is an XPU-2
  handle; the host writes on its own thread and drops the reference");
* the fence, proven load-bearing by a deterministic stress test rather than a hopeful one;
* two real consumers, so it is not dead code waiting for a future release to validate it.

Shipping the mechanism a release before the workload that needs it is deliberate: the
alternative is landing the concurrency and the consumer together and having no way to tell
which one broke.

## 5. The exit-gate test, and the two bugs in its drafts

The release gate asks for "egress fences proven by a stress test that consumes frames from the
wrong side on purpose". Three things had to be true for that test to mean anything, and the
first two drafts each missed one:

1. **It must be deterministic.** A race caught by luck yields a test that passes on a fast box
   and proves nothing. Heavy GPU work is enqueued *ahead* of the copy, so the copy provably
   cannot have started when `egress` returns — `is_ready()` is asserted `False`.
2. **It must not be satisfiable by the allocator.** Draft one compared an unfenced read against
   the truth and **they matched** — not because the copy had landed (the event said it had not)
   but because torch's caching host allocator recycled a pinned block still holding an identical
   frame from an earlier row in the same file. Frame content is now tagged uniquely per row.
   *A test that can be satisfied by the allocator is not testing the fence.*
3. **The FENCED read must happen while the copy is still in flight.** Draft two read
   `unsafe_buffer()` first and only then called `tensor()` — by which time the copy had landed
   anyway, so the mutation "make `tensor()` return `self._host` without waiting" **SURVIVED the
   mutation check**. The row now issues two independent handles behind ballast: one is read
   through `tensor()` immediately (which must be exact — impossible without the fence), the
   other through `unsafe_buffer()` (which must be wrong).

Both halves are asserted. Remove the fence and the first fails; remove the ballast and the
second does.

**What the stress rows do NOT cover, said here so the coverage is not overstated.** They pin
the *fence*: one handle read through `tensor()` while the copy is in flight, one through
`unsafe_buffer()`. They do not pin *concurrency between handles* — N threads each issuing an
egress and racing to fence — because on one DMA engine per direction those copies serialise, so
such a row would assert scheduling the hardware already guarantees and would fail for
timing reasons on a busier box. The concurrency that IS in this release lives in the two
consumers, and it is pinned there instead, in `tests/test_v0331_audit.py` (the demote drain's
identity-and-device commit re-check) and `tests/test_v0332_audit.py` (spill write ordering).
When v0.34's async-write path gives a handle to a thread that outlives the caller, that is the
release where a multi-handle stress row starts asserting something real.

---

## 6. Two decisions this doc owed an argument for

**The copy is issued on the CURRENT stream, not on a dedicated copy stream.** A dedicated
stream is the textbook shape and it is deliberately not what this does. It would require
cross-stream ordering in both directions — record on the compute stream, wait on the copy
stream, and tell the caching allocator about the borrow (`record_stream`) so the source block
is not recycled into another stream's allocation while the DMA is still reading it. Getting
that wrong produces a use-after-free that appears only under memory pressure, which is the
worst-shaped bug this project can ship. Against that: §4 measures the asynchrony itself at
**1.01–1.06×**, i.e. noise. Buying a class of allocator bug for noise is a bad trade. The
revisit gate is concrete — a consumer that genuinely has GPU work to overlap the copy against
(v0.34's async-write path), at which point the copy stream earns the ordering it costs.

**The handle holds its SOURCE, not just its destination, until the fence.** Owning the
destination is the obvious half; owning `_src` is the half a reader asks about, because the
caller usually still holds the source anyway. It is held because "usually" is not a lifetime
rule: a caller that drops its last reference the moment `egress` returns hands the block back
to the caching allocator, and the DMA is then reading a buffer the allocator considers free.
One reference for the duration of a copy is the cheapest possible fix. It is released in
`wait()` — at fence time, not at handle death — so VRAM comes back at the earliest moment it
provably can, and `_event is None` means "fenced" with no second flag to disagree with it.

**The per-call cost, measured, since "bounded" is not a number.** On the paths where `egress`
returns an already-complete handle (a CPU source — i.e. every default ComfyUI cook), it costs
**20.0 µs against 16.3 µs** for the bare `empty()+copy_` it replaced at 512², and
**1876 µs against 1769 µs** at 2048² — a ~4 µs constant plus the same copy, on a `put()` that
is itself 1123 µs. On a CUDA source it is **360.7 µs against 348.5 µs** (512²) and
**5129 µs against 5117 µs** (2048²) versus a pinned blocking copy, fence included. So the
machinery is a low-microsecond constant everywhere and the §4 ratios are the whole story;
there is no hidden per-call tax being waived.
