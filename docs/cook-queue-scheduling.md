# Cook-queue scheduling — design note (SCHED-4, v0.31.0)

The engine can already be *stopped*. SCHED-3 wired a `CancelToken` through every cook
so a host's Stop button reaches a running tier, a tiled strip loop, and the
interpreter's statement walk. This note is the generalization the compositor reports
ask for: **stop, but for a reason other than the user giving up** — "something more
important arrived."

It records what v0.31 ships, the measurements behind the shape it took, and the parts
deliberately not built.

---

## 1. The problem

A compositor is never idle. While the user drags a slider (interactive), the engine
would also like to be: building CACHE-7 checkpoints, pre-rendering the next few frames
of a scrub, and cooking the branch a panel is about to open. All of that is useful and
none of it may cost the slider a single frame.

Two things are needed and neither exists today:

1. **A queue** — somewhere for non-interactive work to wait, with a policy for which
   waiting job runs next.
2. **Preemption** — a running background cook must yield to an interactive one
   *promptly*, and then not be thrown away.

Everything below is engine work by §0's admission test: a host cannot build priority
admission or mid-cook preemption on published APIs without reaching inside the cook.

---

## 2. What already exists (and what that leaves as the work)

This is the load-bearing finding of the design phase, so it is stated first:
**the preemption points already exist.** SCHED-3 put a `_cancel_check(token)` at

| Yield point | Where | Granularity |
|---|---|---|
| A — before the cook starts | `tex_engine.run` | per cook |
| C — a tier failed, before fallback | `tex_engine._interp_fallback` | per tier attempt |
| D — before an fp16→fp32 re-cook | `tex_engine` precision net | per re-cook |
| E — before a tiled OOM re-cook | `tex_engine` OOM ladder | per re-cook |
| F — between tiles / batch strips / halo strips | `tex_memory` | **per strip** |
| — per top-level statement | `tex_runtime/interpreter` | **per statement** |

Two of those are exactly the boundaries the roadmap names. "Tile-loop iterations" is
yield F. "Fused-stage boundaries" needs one observation: `compile_fused` splices a
chain into **one** program whose statements carry `loc.stage` (Q-4), and the
interpreter polls the token *per top-level statement* — so every fused-stage boundary
is already a poll site, and then some. There is no new instrumentation to add.

So SCHED-4 is **not** "add preemption to the engine." It is: mint the token, decide
who gets to trip it, and decide what happens to the job that was tripped. That is a
queue, and it is the whole item.

Also already present, and reused rather than re-invented:

- **ENG-9 per-thread interpreters** — `_get_interpreter()` is thread-local, so a
  worker thread cooking off the main thread is already sound.
- **The background compile pool's thread discipline** (`compiled._COMPILE_POOL`, a
  `max_workers=1` executor) — the precedent that engine background work is a *single*
  serialized worker, not a pool. §5 argues that is right here too.
- **`CookCancelled`** — deliberately unrelated to any OOM type, so the OOM ladder can
  never mistake a preemption for a recoverable failure and silently retry it.

---

## 3. Three admission classes, not two

The roadmap is explicit that two tiers is the wrong count, and the reason is a real
workflow: a user starts a render and *keeps working*. That render is background — it
must not stall the slider — but it is also **committed**: the user asked for it by
name, and shedding it to make room for a speculative prefetch would be a bug the user
experiences as "my render vanished."

| Class | Preempts | Preempted by | Shed under pressure |
|---|---|---|---|
| `INTERACTIVE` (Tier A) | everything below | nothing | never |
| `COMMITTED` (background) | `SPECULATIVE` | `INTERACTIVE` | **never** |
| `SPECULATIVE` (Tier B) | nothing | both | **first** |

The distinction that matters is on the last two columns, and it is why two classes
cannot express the workload: `COMMITTED` and `SPECULATIVE` are identical in priority
against Tier A (both pause) and opposite in durability (one is never dropped, the
other is dropped first). A single "background" tier has to pick one, and either
choice is wrong for half the work.

**Shedding is a distinct outcome from preemption**, and the API keeps them distinct:

- *Preempt* → the job returns to the queue at the head of its class. Transient. The
  host is not told; from its side the job is still pending.
- *Shed* → the job is finished as `CANCELLED` and its waiter is woken. Terminal.

Collapsing the two — the obvious implementation, one `cancelled` flag — is what makes
a scheduler silently lose work.

---

## 4. Re-queue: never abandon a result the cache could still use

Two failure modes here, and the second is the subtle one.

**(a) A preempted cook loses its partial work — and that is not a small thing.** A TEX
cook is atomic: there is no "resume from statement 7." A preempted job re-cooks from
the top.

The first version of this note called that "accepted, not fixed" and claimed CACHE-2
would make the retry cheaper. **Both halves were wrong**, and an audit measured it: a
cook that never finishes never populates a cache, so there is nothing to hit on the
retry, and a 460 ms `COMMITTED` render against a 5 Hz slider made **156 attempts and
zero completions in 25 seconds** — 22.9 s of CPU for nothing, 73.8× the cost of the
render. The rule is simply:

> `T_render > mean gap between interactive requests` ⇒ **the render never completes.**

Preemption without a floor is not "the render pauses", it is "the render restarts".

So the queue carries a **starvation brake**, in two parts (`§tex_cookqueue`):

- `min_quantum_ms` — a running job is not interrupted during its first quantum. Enforced
  by the *token*, not at the submit site: refusing to raise the flag would DROP the
  request rather than defer it, and nothing would re-raise it. Setting it immediately
  and honouring it at the first yield *after* the quantum bounds Tier A's wait to
  `quantum + one yield`, with no timer and no lost request.
- `max_preemptions` — after this many, a job runs to completion. This is what turns
  "never finishes" into "finishes, after at most one bounded wait". It is a genuine
  refusal rather than a deferral: there is no later moment at which preempting a
  budget-exhausted job becomes right.

The same render now completes in 1.2 s after 4 attempts. §8 records what the brake
costs Tier A, because it does cost it something.

Resumable cooks are what would remove the trade entirely, and the measurement above is
the reopen gate §9 asked for — met, but not by this release.

**(b) A cook that already finished must not be discarded because a preempt raced it.**
This is the clause the roadmap phrase "never abandon a Tier-B result the cache could
still use" is pointing at. Preemption is cooperative: the flag is set, and the cook
notices it *at its next yield point*. If the cook returns first, the result is real,
correct, and paid for. The rule:

> **The preempt flag is consulted only by the token, never by the worker.** If
> `_run()` returns a value, the job completes with that value — whatever flags were
> raised while it ran.

Written the other way round (worker checks the flag after the cook, discards if set)
the scheduler throws away completed frames under exactly the load where frames are
most expensive. It is a one-line difference and it is the correctness heart of §4.

---

## 5. Threading: one worker, not a pool

The queue runs **a single worker thread**, and this is a deliberate rejection of the
concurrent-cook design.

The reasoning is the GIL plus what a TEX cook actually is. The tree-walking
interpreter is Python-heavy — the dispatch, the coordinate builtins, the per-statement
walk all hold the GIL — and only the torch kernels release it. Running Tier B
concurrently with Tier A therefore does not overlap two cooks; it **interleaves two
Python interpreters onto one core and slows Tier A down**, which is the one outcome
the whole item exists to prevent. A second worker would buy overlap only for programs
that are almost entirely kernel time, and those are exactly the programs whose Tier-A
latency is already dominated by the GPU.

With one worker, "preemption" has a precise and cheap meaning: trip the running job's
token, let it unwind to the worker loop, push it back, pop the higher-class job. No
locking of engine state, no concurrent access to the per-thread interpreter, no
question about which cook owns the CUDA stream.

The consequence to state honestly: **Tier A waits for the running job to reach its
next yield point.** That interval is PM-7's measurement, and it is the design's real
cost. §8 records what it measured.

GRAPH-2 (a parallel multi-region executor) is the item that would revisit this; it
needs the MUT-cache sharding ENG-9 already flags, and it is not this release.

---

## 6. The queue is not on the default path

Invariant #7 applies to the *queue itself* (doc 39 §8 says so explicitly). The
posture is the one ROI-3, CACHE-2 and SCHED-2 already established:

- `tex_cookqueue` is a module a host **constructs**. Nothing in `tex_node.py`,
  `tex_engine.cook()`, or the ComfyUI path imports it or reaches it.
- No worker thread exists until `CookQueue()` is instantiated, and none is started
  until the first `submit()`.
- The engine gains **no new code on the cook path**. The queue passes its token
  through the existing `cancel=` parameter; from `tex_engine`'s side a queued cook is
  indistinguishable from a SCHED-3 cook with a Stop button attached.

That last point is what makes the invariant provable rather than asserted: the diff
to the cook path is empty.

---

## 7. What PROF-1 and PRED-1 add on top

Recorded here because the three items were designed together.

- **PROF-1** is the cost oracle. The queue does not time cooks itself; it asks
  `tex_runtime.profile.predict(...)` and gets an EWMA over past cooks of the same
  (program, device, precision, resolution-bucket) key.
- **PRED-1** is the admission policy for `SPECULATIVE` only. A speculative submit
  carries a `confidence` and a `reason`; the queue scores it and may refuse it
  outright, order it against its peers, or shed it later. `INTERACTIVE` and
  `COMMITTED` are never scored — they were asked for.

Both are separate items with their own notes in the CHANGELOG; the seam between them
and the queue is two function calls.

---

## 8. Measured (this box: RTX 2080 SUPER, sm_75, no Triton)

PM-7 asks: an interactive request preempts a running Tier-B cook and reaches first
kernel within a measured bound — target ≤ 1 stage/tile boundary, single-digit ms at
proxy resolution. `benchmarks/cookqueue_bench.py`, 512², one long speculative cook
running plus a backlog:

| | CUDA | CPU |
|---|---|---|
| solo cook, no queue | 0.336 ms | 0.671 ms |
| through an **idle** queue | 0.402 ms (tax +0.066) | 0.830 ms (tax +0.159) |
| **submit → tier start, under load** | **0.595 ms** | **0.804 ms** |
| submit → first executed statement | 0.886 ms | 1.741 ms |
| speculative: submitted / resumed to DONE | 27 / **27** | 27 / **27** |

An order of magnitude inside the bound, and §4's obligation met: nothing abandoned.
The yield latency §5 warned about is the `submit → tier start` row.

**What the brake costs, stated plainly.** Without it that row was 0.193 ms (CUDA) /
0.506 ms (CPU) — and background renders never finished. The quantum lets a job already
inside it complete rather than be restarted, so Tier A sometimes waits out a short cook.
That is a different contract, not a regression against the old one, and the old one was
not shippable.

The honest caveat is the one PM-6 carries too: PM-7's target names the sm_120 box;
this is not that box, so these are this hardware's numbers, not a restatement of the
target.

---

## 9. Deferred, with the gate that would reopen each

- **Resumable cooks** (checkpoint mid-program, resume after preemption). Needs a
  serializable interpreter state, which the tree-walker does not have and codegen
  actively does not (its locals are Python frame state). **The reopen gate is MET**:
  §4a measured a render whose cost was entirely re-cook (156 attempts, 0 completions).
  The starvation brake makes the current design correct rather than optimal — a long
  render still restarts up to `max_preemptions` times — so this is now a live item with
  a measured motivation, not a hypothetical.
- **A worker pool** (§5). Reopens with GRAPH-2's parallel region executor.
- **Sub-statement preemption.** The floor today is one top-level statement / one
  strip. A single statement over a 4K frame is a real interval. Reopens if PM-7's
  yield latency exceeds the single-digit-ms target on realistic programs; the fix
  would be a poll inside the tiling loop's inner kernel dispatch, not a finer
  interpreter walk.
- **Aging beyond a budget.** `max_preemptions` is a counter, not a priority decay: a
  job's Nth preemption costs the same as its first until the budget runs out. A real
  aging scheme (each preemption raises the class a job must yield to) would smooth the
  worst case rather than cap it. Reopens if a host reports Tier-A latency spikes from
  budget-exhausted jobs.
- **Priority inheritance.** If a Tier-A job needs a boundary a `SPECULATIVE` job is
  currently cooking, the right move is to promote the speculative job rather than
  preempt it. There is no dependency edge between queued jobs today (the host owns
  the graph), so there is nothing to inherit *from*. Reopens with GRAPH-1.
