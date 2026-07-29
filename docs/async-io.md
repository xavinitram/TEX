# IO-1 — async source materialization, and the async-write contract

*Design doc for the v0.34 "Sources & sinks" items §3.2 (IO-1, L) and §3.3 (async writes,
S — contract only). Written before the code. DATA-7 (`docs/frame-providers.md`) is the
substrate: this document is about **when a cook may start**, not about where pixels come
from.*

---

## 1. What is actually new

Doc 41 is precise that almost everything here is templated, and names the two mechanisms
that are not:

**(a) A value-less binding in an engine that derives identity from binding *values*.**
`prepare()` builds `binding_types` by calling `infer_binding_type` on every value
(`tex_engine.py:1104`), and `infer_binding_type` ends in `return TEXType.FLOAT`
(`tex_marshalling.py:330`) — a silent catch-all. A promise wired today does not fail; it
types as FLOAT, mints a fingerprint for a program that does not exist, and compiles
something wrong. Identity corruption, not an error.

**(b) Dependency-aware admission in a queue whose jobs are opaque closures.** A `Job` is
`fn(cancel_token)`, eligible the moment it is submitted (`tex_cookqueue.py:288-314`). There
is one worker and no WAITING state, and the one-worker contract is load-bearing (module
docstring §2: a second worker interleaves two Python interpreters onto one core). So the
worker must never block on I/O — which means the *queue*, not the job, has to know that a
job is not ready.

Everything else — the speculative class, the shed policy, the refusals ledger, the
`PREFETCH` reason constant reserved since v0.31 (`tex_cookqueue.py:735`) — already ships.

---

## 2. The central decision: readiness vs identity

Doc 41 poses it as a fork. **Decision: `Promise` declares its type, shape and device up
front (the first option). The host-side gate object is not shipped.**

The argument for the fallback is real — it needs no engine change at all. It is refused
because it does not solve (a). A host gating `submit()` behind its own readiness check
still eventually calls `tex_engine.cook(bindings)`, and the day someone passes an unlanded
object, `infer_binding_type` types it FLOAT. The gate object moves the hazard; the
declaration removes it.

```python
p = Promise("A", type=TEXType.VEC4, shape=(1, 1080, 1920, 4), device="cuda:0")
job = queue.submit(fn, klass=COMMITTED, inputs=[p])
...                                      # elsewhere, on the host's I/O thread
p.land(tensor)                           # validated, then the job becomes eligible
```

Three consequences, each pinned:

* **Identity is computable before the value lands.** `infer_binding_type(promise)` returns
  the declared type, so a host can fingerprint, score and key a job whose pixels are still
  on disk.
* **The landed tensor is validated against the declaration** at the marshalling seam, and
  a mismatch is E7006 naming both shapes. A promise that lies is a loud failure at the
  moment of landing, not a wrong cook later.
* **`infer_binding_type` learns to refuse.** Anything that is not a recognized shape now
  raises E7005 instead of returning FLOAT. Doc 41 is explicit that this fix ships even if
  IO-1 slips, and it is the half of the item with real blast radius — so it lands with the
  suite as its own evidence, and any type the tree genuinely relies on gets an explicit
  branch and a comment, never a silent fallback.

### The engine seam, and its invariant-#7 cost

`prepare()` resolves promises to tensors before inferring types, so **the interpreter,
codegen and every tier see ordinary tensors** — the same load-bearing simplification
DATA-6 uses for planes. The cost on the default path is one class check per binding,
guarded so the resolution loop does not run at all when no promise is present. That is
measured against a null control like every other invariant-#7 claim; a structural argument
is not enough here, because this is the one edit in the release that touches `prepare()`.

An unlanded promise reaching `prepare()` is E7007. It means a host cooked around the
queue's admission, and silently blocking the cook thread on an I/O wait is exactly the
failure the WAITING state exists to prevent — so it is refused rather than waited on.

---

## 3. Admission: the WAITING state

**Decision: a waiting job lives in its class deque, marked `WAITING`, and
`_next_locked` scans past it. It is NOT parked in a side table.**

The side table is the obvious implementation and is cheaper by one linear scan. It is
refused because `Job` objects are reached by five paths — `_next_locked`, the shed loop,
`cancel()`, `close()`'s drain, and `snapshot()` — and a second home means five places that
must remember to look in both. The shed path is the one that matters: a waiting
SPECULATIVE prefetch is *exactly* what backpressure needs to drop, and a side table hides
it from the policy that exists to drop it.

The cost is honest: `_next_locked` becomes a scan for the first PENDING job instead of a
`popleft`. It stops at index 0 whenever nothing is waiting (every cook in the tree today),
and `_enqueue_locked` already does a linear insert per speculative submit, so the queue's
per-job complexity is unchanged.

The rest follows:

* `submit(..., inputs=[p, …])` — a job with any unlanded input starts `WAITING`.
* **A WAITING submit never preempts.** `_preempt_for_locked` is called only for a job that
  can actually run. Tripping a running render for a job that is still waiting on a disk
  read is the worst possible trade: the render restarts (§4a — there is no resume) and the
  preemptor cannot even start.
* Each promise gets a landing callback that, under the queue's lock, flips the job to
  PENDING once *all* its inputs have landed, and notifies. The callback does no I/O and
  runs on the host's thread, not the worker's.
* A promise that **fails** fails its jobs: they finish FAILED with the provider's error,
  rather than waiting forever.
* Per-branch granularity falls out with no work: the unfused host already submits
  per-stage jobs, so a branch whose inputs landed runs while its sibling waits. A fused
  chain is one job by construction. The test asserts cook *order*, not timing.

---

## 4. Prefetch, and the PROF-1 pollution guard

```python
tex_provider.declare_window(queue, source_key, t0, t1, *, confidence=0.5, mode="sample")
```

mints one SPECULATIVE job per quantized frame in `[t0, t1]`, reason `PREFETCH`, each of
which fetches into the media pool. They are priced, ordered and shed by the existing
`SpeculativePolicy` — a prefetch window is a bet like any other, and the one that arrives
during a render loses to the render by the rules already in the tree.

**The pollution guard is structural, not incidental.** Doc 41: an I/O wait recorded as
compute cost would poison CACHE-7 placement and PRED-1 admission in one stroke. The
existing feedback is already gated on `profile_key is not None`, and prefetch jobs pass
none — but "we happened not to pass a key" is not a guard, because the host controls that
argument. So `Job` grows `feeds_profile` (default True), `declare_window` sets it False,
and the feedback checks it. The pinned test submits a deliberately slow prefetch *with* a
profile key and asserts the table is unchanged.

**Cancellation is drop-on-landing.** A shed or cancelled prefetch that is already inside
the provider's read cannot be stopped from outside; what TEX guarantees is that its result
is never installed — the job re-checks its token immediately before the pool insert. A
provider that wants to do better may accept the cancel token and poll it mid-read; that is
optional by contract, and a provider that ignores it is correct, just slower.

**Backpressure is the pool refusing an insert.** A speculative insert into a `MediaCache`
that is already at budget is refused rather than admitted-then-evicting: a prefetch is a
guess, and evicting a frame someone actually asked for to make room for a guess inverts
the whole policy. Refusal is cheap precisely because the host can fetch on demand — the
frame is still on disk.

---

## 5. Async writes (§3.3) — a test, not a subsystem

The contract in one line: **a completed frame is an XPU-2 `FrameHandle`; the host writes
on its own thread and drops the reference; TEX's obligation ends at "the next cook starts
immediately".**

There is nothing to build. `FrameHandle` (`tex_runtime/streams.py:50`) already carries the
fence, already keeps the source alive until the copy lands, and already degrades to an
already-complete handle on CPU. What is missing is a *consumer* and a *proof*, and doc 41
asks for exactly those two:

* **the scheduling half** — through the CookQueue: complete frame N, hand its handle to a
  deliberately slow writer thread, submit N+1, and assert N+1's first statement executes
  while the writer still holds N's handle;
* **the fence half** — the writer's bytes are bit-exact against a synchronous write, using
  the XPU-2 stress harness.

The two reference consumers adopt the handle: `tex_cli`'s save path and `host_demo`'s
`rgba_bytes` blit. Both accept a handle *or* a tensor, because a CPU cook hands back a
tensor and forcing every caller through a handle would be ceremony on the path that has
nothing to overlap.

---

## 6. PM-9, and what the number will and will not mean

The v0.34 exit: headless 100-frame playback through the CookQueue where input prefetch and
output writes overlap compute, **≥ 80% of I/O hidden** against the serial baseline.

The harness is a synthetic slow provider (a fixed sleep per fetch — deterministic, no real
disk) plus a synthetic slow writer. The number to report is
`1 − (t_overlapped − t_compute) / t_io_serial`: of the I/O time a serial loop would have
paid, how much disappeared behind compute.

Stated up front so the result cannot be over-read: with synthetic sleeps this measures
**the scheduler**, not a disk. A sleep releases the GIL cleanly and a real decode does
not — a Python-side decoder would contend with the single cook worker in a way this
harness cannot see. That is the honest caveat the number ships with, alongside the
standing sm_75 hardware caveat.

### The result (`benchmarks/io_playback_bench.py`, 100 frames, 512², sm_75)

| grades/frame | compute | serial | overlapped | hidden | speedup |
|---|---|---|---|---|---|
| 24 | 410 ms | 1374 ms | 465 ms | **94.3%** | 2.95× |
| 48 | 803 ms | 1762 ms | 844 ms | **95.7%** | 2.09× |
| 1  | 35 ms  | 980 ms  | 462 ms | 54.9% | 2.12× |

**PASS**, and the third row is the boundary rather than a failure. "I/O hidden behind
compute" presumes there is compute to hide behind: at one grade per frame the cook is
0.35 ms against ~9.5 ms of I/O — 3.5% of the work — and no scheduler can hide 9.5 ms behind
0.35 ms. That run is already at its three-way overlap floor (fetch 400 ms ∥ write 400 ms ∥
cook 35 ms ⇒ ~400 ms + ramp, measured 462 ms); what it overlaps is I/O with I/O, which is
what the 2.12× is. The 24-grade row is the realistic ratio and the one PM-9 is claimed on.

### The finding worth keeping: Tier-B prefetch does not hide I/O under load

`declare_window` prefetch riding the same queue as the cooks measured **1332 ms against
1374 ms serial — ~3% hidden.** That is not a bug, it is the one-worker contract doing what it
says: a SPECULATIVE prefetch occupying the single worker is *taking turns* with compute, not
overlapping it. Tier-B prefetch is for a worker that is otherwise idle — priced, sheddable,
ledgered. The **promise path** is what hides I/O while the queue is busy, because the fetch
happens on the host's own thread and only the *landing* touches the queue. Both modes ship;
this is the sentence that says which is which.

---

## 7. Definition of done

1. `Promise` with declared type/shape/device; `land()` / `fail()`; landing callbacks.
2. `infer_binding_type` refuses unknown objects (E7005) and types a promise by its
   declaration.
3. Marshalling-seam validation of a landed tensor (E7006) and refusal of an unlanded one
   at `prepare()` (E7007).
4. CookQueue WAITING state: `inputs=`, scan-past admission, landing wake, no preempt on a
   waiting submit, waiting jobs visible to shed / cancel / close / snapshot.
5. `declare_window` prefetch through `SpeculativePolicy`, with `feeds_profile=False` and
   the pinned pollution guard.
6. Cancellation (drop-on-landing) and backpressure (refused speculative insert) rows.
7. Per-branch streaming proven by cook order with a deliberately slow provider.
8. The async-write scheduling half and fence half; `tex_cli` + `host_demo` adopt the
   handle.
9. Invariant #7 on the queue itself (the WAITING state adds admission-time work only) and
   on `prepare()`, both null-controlled.
