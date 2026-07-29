# DATA-7 — FrameProvider: the host source protocol

*Design doc for the v0.34 "Sources & sinks" item. Written before the code, per §10.1's
design-doc-first rule for L items. Doc 41 §3.1 is the brief; this document makes the
decisions it names, with the tension stated for each, and is the contract the tests pin.*

---

## 1. The problem

TEX can read any pixels a host hands it as a binding, and nothing else. Every temporal
op in the language — `fetch_frame`, `sample_frame`, the 3-arg `@A[x,y,f]` sugar — indexes
**inside the batch that was already marshalled**. ROI-6's `frame_window` analysis
(`tex_roi.py:640`) stops at the batch edge for exactly that reason: there is nothing past
it to reach.

Motion blur over a shutter, a temporal median over ±3 frames, a flow-warp from the
previous frame: all of them want frames the batch does not contain, and all of them are
the same request — *give me this source at this time*. Marshalling the whole range as one
giant batch is the workaround, and it is the wrong shape: it pays for every frame up
front, at cook resolution, in the cook's memory budget, whether the program reads it or
not.

DATA-7 is the seam that answers the request instead: a **host source protocol**, a
governed pool that remembers what it fetched, and two stdlib functions that read through
both. It is also the substrate IO-1 (§3.2) plugs async file I/O into — which is why the
protocol is a *synchronous* function returning a tensor, and readiness is IO-1's problem,
not this item's.

---

## 2. Where the protocol lives

**Decision: a new top-level `tex_provider.py`, not a second protocol inside
`tex_runtime/host.py`.**

The tension is real and doc 41 words it as "a `FrameProvider` Protocol beside
HostServices", which reads like the same file. Two reasons it is not:

* `tex_runtime/host.py` is the **PORT-1 boundary** — the one place `comfy.model_management`
  is imported, pinned there forever by `test_port1_import_lint`. Its whole value is that
  the grep says one thing. A frame cache, a governor registration and a reattach line in
  that file make the lint's subject ambiguous.
* the protocol is the small half. The pool — byte accounting, LRU, per-device buckets,
  version-keyed invalidation, governor arbitration — is engine state of the same weight as
  `tex_results.ResultCache`, and this repo gives that one module apiece.

What *is* copied from HostServices, deliberately and line for line, is the **shape**:
a `Protocol`, a `Null` default that is a real implementation rather than a stub, a
process-wide `get_provider()` memo, and `set_provider()` / `reset_providers()` for tests
and for a non-ComfyUI host wiring itself in. A reader who knows `host.py` knows this file.

The name `tex_provider.py` is not invented here — `docs/roadmap.md` §9 already pencils it
in the v0.34 row. The roadmap of record wins over a fresh preference.

---

## 3. The protocol

```python
class FrameProvider(Protocol):
    provider_id: str                                  # stable across a session; keys the pool
    def fetch_time(self, source_key: str, t: float) -> torch.Tensor: ...
    def sample_time(self, source_key: str, t: float) -> torch.Tensor: ...
    def quantize_time(self, source_key: str, t: float) -> float: ...
    def source_version(self, source_key: str) -> int: ...
```

`fetch_time` returns the source's frame **at or nearest to** `t`; `sample_time` may
interpolate between the two frames bracketing `t`. Both return `[1,H,W,C]` (or `[H,W,C]`,
normalized on ingest) at the source's own resolution — **not** the cook's. Host owns
decoding, colour, and everything else about how the pixels came to exist.

The split between the two methods is the host's to honour and TEX does not check it: a
provider that returns the same pixels from both is a legal nearest-neighbour provider.
What TEX guarantees is that the two are **cached separately** — a `sample_time` result is
never served to a `fetch_time` call, because a provider is allowed to make them differ.

### `quantize_time` — why the protocol owns it

Doc 41: *"`sample_time` defining time quantization so 23.999999 vs 24.0 doesn't
double-cache."* Only the provider knows its own frame rate, so only the provider can say
which two times are the same frame. `NullFrameProvider` and the shipped
`SyntheticFrameProvider` quantize as `round(t * rate) / rate` with a declared `rate`;
`rate = 0` means "do not quantize", and a host with variable-rate media returns whatever
its index says.

The engine quantizes **once**, at the pool boundary, and passes the quantized value on to
`fetch_time`/`sample_time`. A provider therefore never sees a time it would have rounded —
which makes the pool key and the fetch argument the same number by construction, rather
than by two implementations agreeing.

---

## 4. The language surface

Two stdlib entries, mirroring their in-batch twins argument for argument:

```
fetch_time (source, t, px, py)  → vec4      // nearest-neighbour, pixel coords
sample_time(source, t, u,  v)   → vec4      // bilinear, normalized coords
```

`source` is a TEX string — normally a literal; any string-valued expression works.
`t` is the source's own time, in whatever units the provider indexes by (seconds or
frames — the provider's choice, stated in its docs, never guessed at here).

**Why not a two-argument form returning "the frame".** `fetch_time("plate", t)` reads
better and is what the protocol's own methods look like. It cannot be the language
surface: TEX has no first-class image value — bindings are the only tensor-shaped thing an
expression can name, and a stdlib function that returned one would need a new TEXType,
which §6 of the language roadmap rejects outright. Taking coordinates makes these two
functions exactly what `fetch`/`sample` already are, with the image argument replaced by
`(source, t)`. Nothing else in the compiler has to learn a new kind of value.

### `t` must be uniform across the grid

**Decision: a `t` that differs pixel-to-pixel is refused, loudly, as E7003.**

The tension is that a per-pixel `t` is the *interesting* generalization — it is a retime
map, and it is exactly what a warp wants. It is also unbounded I/O from a single call: one
`sample_time` over a smooth ramp asks for as many source frames as there are distinct
values, at cook resolution, inside one statement. There is no honest cap to pick, and the
failure mode of guessing one (silently servicing 8 of 4096 requested frames) is wrong
pixels.

So v1 checks and refuses. A scalar or 0-dim `t` takes a fast path; anything else is
compared against its own first element, and the refusal message names how many distinct
values it saw and the two ways out (hoist the time out of the grid, or cook one frame per
playhead). The check costs one reduction on a call that is about to do file I/O.

The same rule is what makes a **per-batch-element** `t` — `time + fi/fps` over a B=100
batch — a refusal rather than a silent single-frame read. That case is bounded (≤ B
frames) and is the natural next step; it is deferred with its gate in DEVELOPMENT.md.
PM-9's playback does not need it: a playback loop cooks one frame per playhead, which is
the shape the CookQueue and `time_context` already have.

### Registry tags, and one correction to the brief

```
spatial=True, sync=True, footprint='image'
```

`spatial=True, footprint='image'` is copied verbatim from `fetch`/`sample`
(`stdlib.py:962,1013`) — an arbitrary-coordinate gather is precisely what `'image'` means.
`sync=True` puts both names in `graphed._SYNC_STDLIB`, so the CUDA-graph tier refuses to
capture a program that calls them (a host callback inside a capture is not a sync, it is a
foreign call, and it would be baked once and replayed forever).

**The correction.** Doc 41 §3.1 asks for `footprint=('frame', i)`-class descriptors, and
`docs/roadmap.md` §9 hedges it as "(or `'image'` if unbounded)". `('frame', i)` is wrong
here, and not merely imprecise: ROI-6's `_frame_ops` (`tex_roi.py:617`) reads argument `i`
of any `('frame', i)` call as a **batch index** and hands it to
`_extract_pixel_offset(arg, "fi")`, so `fetch_time("plate", fi - 1, …)` would be recorded
as "this program reads batch frame fi-1". It reads no batch frame at all. The window would
be a lie that the strip planner acts on.

`'image'` says the true thing — one output pixel may read anywhere in an input the ROI
planner does not model — and both derivations the DoD asks for follow from it with no new
machinery: `non_local` (footprint ≠ `'point'`) refuses M-4 strip tiling, and `sync=True`
refuses graph capture. `batch_sliceable` is separately unaffected: it already returns
False for any program the strip planner cannot model.

---

## 5. The media pool

A `MediaCache` in `tex_provider.py`, keyed

```
(provider_id, source_key, mode, quantized_t, version)
```

where `mode` is `"fetch"` or `"sample"` (§3) and `version` is
`provider.source_version(source_key)` at insert time.

* **No source key → no caching, ever.** CACHE-6's precedent, restated: an empty
  `source_key` fetches every time. A host that cannot name a source cannot promise
  anything about it.
* **LRU with per-device byte buckets**, the same `governed_bytes(dev_type)` /
  `evict_bytes(need, dev_type, playhead)` pair `ResultCache` exposes — so it registers
  into `CacheRegistry` with the existing call and no new governor concept.
* **In-memory only. There is no disk tier, and that is a decision, not an omission.**
  CACHE-8's ladder exists because a cooked frame is expensive and has nowhere else to
  live. A source frame's disk tier is *the source file*, which the host already has and
  can already decode. Spilling a decoded copy beside it buys a faster decode at the price
  of a second copy of the user's media on their disk.

### Where it sits in the eviction ladder

`evict_order=40` — after `stdlib` (10), **before** `results` (50), before `graphs` (90).

A media frame rebuilds with one host fetch. A result frame rebuilds with a cook, and that
cook may itself have to fetch sources. Strictly cheaper to rebuild ⇒ drained first, which
is the ladder's whole ordering rule. The number is an argument to the register call, so a
host whose provider is a network mount can reverse it.

---

## 6. Purity, invalidation, and the honest gap

**The contract:** a provider must be a pure function of `(source_key, quantized_t)` for as
long as `source_version(source_key)` does not change. TEX never stats a file, never
watches a directory, and never guesses — the host knows when it re-exported, so the host
bumps.

`bump_source_version(source_key)` does two things: increments the stamp, and drops that
source's pool entries. Dropping is not strictly required (the version is in the key, so
old entries are unreachable) but leaving them would let a re-export leak the whole
previous version's bytes into the governor's accounting until pressure evicted them.

**The gap, stated plainly.** The *result* cache is keyed by `lineage_key`, which knows
nothing about sources. A cook that read `plate@v3` and a cook that read `plate@v4` mint
the same result key, and the second is served the first's pixels.

The fix that would close it structurally — the engine folding the read-set into every
lineage key — needs the read-set **before** the cook, and the read-set is only known
after: the source key can be a `$param`, a string expression, or a loop variable. Deriving
it from the AST closes the common case and silently misses the rest, which is the worst of
both.

So v1 ships the host obligation, spelled and testable:

```python
key = lineage_key(..., flags=tex_provider.source_flags("plate", "matte"))
```

`source_flags` returns `("src=plate@3", "src=matte@1")`, which `lineage_key` already folds
through its existing `flags` component — no key-shape change, so no cache-wide churn for
hosts that never call it. A test pins that a version bump changes the flags and therefore
the key. The automatic derivation is a recorded deferral with its gate.

---

## 7. Errors: the E7xxx family

A new family, per R8 — stable anchors, rendered into `Error-Codes.md` by the generator:

| code | meaning |
|------|---------|
| E7001 | no frame provider is registered (the Null default refusing) |
| E7002 | the provider raised while fetching — names the source and the time |
| E7003 | a per-pixel / per-batch `t` (§4) |
| E7004 | the provider returned something that is not a `[1,H,W,C]` frame |
| E7005 | a binding whose type cannot be inferred (IO-1's loud refusal) |

**Severity is class-dependent, and the provider does not know its job's class** — so the
wrap happens where the class is known, which is the CookQueue. A failed fetch inside an
INTERACTIVE or COMMITTED job raises E7xxx to the waiter, unchanged. Inside a SPECULATIVE
job it is recorded in the policy's refusals ledger and the job ends CANCELLED, because
speculation that alarms is worse than speculation that quietly did not happen. This is the
only queue change DATA-7 makes, and it is six lines in the existing `except` arm.

---

## 8. Reattach

Providers are host-side and process-global; DATA-4 phase 2 (cross-process engine state) is
still deferred, so `reattach()` cannot restore one. The contract states it: **a host must
re-register its providers and re-declare its prefetch windows after a reattach.**

`reattach()` grows a `media_frames` / `media_bytes` line so a host can see what it has —
after a restart that is `0`, and printing the zero is the point: a host that thinks its
pool survived and finds it did not currently has no way to tell.

---

## 9. Why no `examples/*.tex` snippet

Doc 40 §2's stdlib checklist asks for an example snippet when a function carries a
workflow, and these two carry the most workflow-shaped feature in the release. They still
get none, for a mechanical reason worth writing down:

`examples/*.tex` **is** the LANG-3 compat corpus's input set (`compat_corpus.py:109`).
Every file there is compiled and hashed on every suite run, with no host and therefore no
provider. A `fetch_time` example can only produce `ERROR:` there — and the corpus test
fails any `ERROR:` outright, so the example would have to be frozen as a permanent error
into the language's own compatibility proof.

The exemplars the DoD asks for — motion blur and temporal median — ship as test fixtures
against the synthetic provider instead, cooking on CPU and CUDA. The workflow is
documented here and in the function reference; the corpus stays a corpus.

---

## 10. Definition of done

1. `FrameProvider` protocol + `NullFrameProvider` + `get/set/reset_provider`, shaped after
   `get_host_services()`.
2. `fetch_time` / `sample_time` through the full doc 40 §2 checklist: impl, registry entry
   with tags and help data, signature, codegen deferral (no `@_emits` handler), TST-2 edge
   rows, regenerated docs.
3. `MediaCache` registered into `CacheRegistry` at `evict_order=40`, reporting
   `governed_bytes`/`evict_bytes`/`stats()`.
4. Motion-blur and temporal-median exemplars cooking through a synthetic provider on CPU
   and CUDA.
5. Derivations pinned: a `fetch_time` program refuses tiling and refuses graph capture.
6. Purity, version invalidation, `source_flags` keying, reattach line, and the five E7xxx
   rows green.
7. Governor arbitration over a pool holding frames, mixed with the result cache.
8. Invariant #7: with no provider registered, nothing is constructed, nothing is
   registered that reports non-zero, and the default cook path is untouched.
