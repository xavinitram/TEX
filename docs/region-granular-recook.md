# Region-granular recook — design note (CACHE-9, v0.32.0)

*Companion to `docs/roi-spatial-laziness.md` (ROI-1..6), which supplies the analysis and the
oracle lane, and to `docs/results-caching.md` (ENG-12/CACHE-1/2), which supplies the frames
being patched. Roadmap: report 39 §v0.32.0.*

The Memory report calls it "spatial invalidation": when an edit's downstream footprint is a
bounded region — a roto point nudge, a paint stroke, a cropped adjustment — recook only
`footprint ⊕ halo` and patch the cached frame, instead of recooking the frame.

---

## 1. The problem, and the honest starting position

**The win already exists, and it is large.** PM-6 measured it in v0.30: a 10-node comp at
1920², scrubbing the terminal node's slider, is **34.0 ms/frame** whole-frame and
**1.5 ms/frame (~22×)** when only the dirty suffix cooks its window and the nine clean
upstream canvases stand.

**It exists in the wrong place.** All of it lives in `examples/host_demo.py::RoiComp` — a
demo. The engine ships `run_roi` (cook a window) and nothing else: no way to compose windows
along a chain, and no way to write a window into a cached frame. So every host that wants
PM-6's number must re-derive two things the demo got wrong before it got them right:

- **The halo composition.** Patching only the requested rect is wrong the moment a downstream
  stage has a halo — it reads a ring of neighbours just outside the patch, and that ring still
  holds pre-edit pixels. The demo's own comment records the damage before it walked the suffix
  backwards growing by each consumer's reach: *"stage-5 sharpen was wrong over 2157 px and
  stage-9 vignette over 3987 px on any upstream edit."*
- **The ownership.** A cook output arrives frozen under ENG-12, and freezing is a **tripwire,
  not a fence** — on torch 2.12 an in-place op on an inference tensor *lands the write* and
  then raises, so the raise is not a rollback. The demo carries an explicit `_owned` flag and
  clones once before the first window write.

Two subtleties that a host is even less likely to get right, both verified here:

- **A non-executable stage must report UNBOUNDED reach, not zero.** `RoiPlan.halo` is `0` when
  `executable` is `False` — but that means "there is a gather, cook the whole frame", the
  opposite of "reads no neighbours". A consumer that trusts the `0` under-grows the upstream
  window and leaves exactly the stale ring. The whitelist posture — unknown → whole image —
  has to be re-inverted by hand at every consumer.
- **`binding_footprints` under-reports through local aliases.** Its own docstring admits it:
  `vec4 x = @A; @OUT = gauss_blur(x, 2)` reports `A: point`. That is safe for `roi_plan`
  (which blocks and falls back), and *fatal* for a new consumer that sizes a region from
  footprints directly. So the composer must go through `roi_plan`, never `binding_footprints`.

**So CACHE-9's v0.32 job is not to find a speedup. It is to move a proven, easy-to-get-wrong
mechanism behind the engine seam, and to gate it with the oracle every ROI feature ships
behind.**

---

## 2. The scope decision, made here rather than buried in the deferred list

The report's phrasing — "when an edit's downstream *footprint* (ROI-2) is a bounded region" —
reads as if it applies to any graph. It does not, and the reason is structural:

**`roi=` is refused on a fused chain** (`tex_engine.py:1261`: `roi is not None and tier_id ==
"default" and not fused_chain and not has_latent_input`). Fusion splices stages behind local
variables, and the reach analysis cannot see through them — `_has_ungrounded_halo` blocks
exactly that shape. So a region recook of a *fused region* is not a caching problem at all; it
is a reach-analysis problem in the ROI/LANG layer.

That splits v0.32 cleanly, and the split also decides which item PM-8 measures:

| | CACHE-7 | CACHE-9 |
|---|---|---|
| Host shape | one **fused** chain (≤ 16 stages — fusion's region cap) | **unfused** per-stage cooks |
| Unit of reuse | a stage-boundary tap in `ResultCache` | a patched frame region in `ResultCache` |
| Reached by | `cook_checkpointed` | `chain_windows` + `patch_region` |

They are complements, not layers: past 16 nodes a comp is entirely unfused (measured: a
50-node linear graph yields **zero** fused regions), which is why PM-8's "50-node comp" is
CACHE-9's measurement and not CACHE-7's.

**v0.32 ships the unfused half.** Region recook of a fused region is deferred with its gate in
§7.

---

## 3. `chain_windows` — the composition, promoted and generalized

```
stage_halo(code, params)            -> int          # reach, or WHOLE_FRAME if not executable
chain_windows(halos, roi, dirty_from) -> list       # the window each stage must cook
```

Walk the suffix **backwards** from the requested output window, growing by each *consumer's*
halo and clamping to the frame — the same `ROI ⊕ halo` composition `run_roi` performs within
one stage, lifted to the chain. Three things this fixes relative to a host doing it itself:

1. **The inversion is inside.** `stage_halo` returns `WHOLE_FRAME` for a non-executable plan,
   so a consumer cannot accidentally trust a `0` that means "unbounded".
2. **A whole-frame window is a whole-frame window.** Once any consumer reports unbounded
   reach, every window at or above it clamps to the full frame — which is correct, and is what
   makes the composer safe to call on programs it cannot narrow.
3. **It is memoizable in one place.** `roi_plan` folds `$param` values, so its own memo keys on
   them; a scrub changes params every frame, and an un-memoized call re-parses inside the
   frame loop (the demo measured **0.431 ms/frame** across a 10-stage comp before memoizing).

`halos` is a plain list of ints rather than a list of programs, deliberately: the engine does
not own the host's notion of "a stage", and a host that already knows its reaches (or wants to
force one) should not have to fake a program to say so. `stage_halo` is the helper that
produces them.

---

## 4. `patch_region` — the write the engine has never had

There is **no partial-write path in the package today**. Verified: exactly three
slice-assignments exist outside tests and examples (`tex_memory.py:799`, `:882`, `:1134`), and
all three write into a `torch.empty()` allocated in the same call and fully filled before
return. That is *assembly*, never a patch.

So the primitive is new, and its whole content is ownership:

```
patch_region(cache, key, patch, window, *, base=None) -> tensor
```

- Read the base frame (the cached one, or an explicit `base`).
- **Copy it.** Never write into the cached master. A cached frame is frozen, and freezing does
  not stop the write — it only reports it afterwards. `ResultCache.get`'s default
  `copy=True` already hands back an owned clone, so the copy is the *existing* contract being
  used correctly rather than a new cost being invented.
- Write `patch` into `window` on the copy.
- `put` the result under a **new key**, leaving the old entry addressable.

**Version-stamped, and why the existing stamp is not enough.** `ResultCache` entries carry
`frame_version(frozen)`, which is `t._version` — and `frame_version` returns a **constant 0**
for any inference (frozen) tensor. Since `put` always freezes, every entry is stamped 0 and
`verify_unmutated` is permanently true for exactly the buffers being patched. So the stamp
cannot express "this frame is a patched descendant of that one". CACHE-9 puts the lineage in
the *key* instead, where CACHE-1 already has a slot for it: the patched frame's key carries
the base key in `upstream` and the window in `canvas`. That is a real provenance chain, it
survives a spill/restore, and it needs no new entry type.

**The cost is honest, and it is not small.** A patch is a full-frame copy plus a windowed
write, so it is a win for a *cache-resident* canvas patched repeatedly and a **loss** for a
frame patched once. Measured on the PM-8 comp (50 nodes, 2048², 512² window):

| | CPU | CUDA |
|---|---|---|
| whole-frame all-dirty | 934 ms | 172 ms |
| **region, all 50 stages dirty** | **4437 ms (0.21×)** | **3980 ms (0.04×)** |
| **region, mid-graph edit** | **156 ms (5.98×)** | **17.4 ms (9.88×)** |

So the rule a host needs, stated plainly: **route an ALL-DIRTY recook whole-frame.** With every
stage dirty, region recook pays 50 full-frame clones (67 MB each at 2048²) *and* cooks
windows that the accumulated halos have already grown most of the way back to the frame — it
is 5–25× SLOWER than simply cooking the frame. The win comes from the clean prefix standing:
with the edit mid-graph, 25 canvases are reused and only the suffix cooks its window.

That asymmetry is why `dirty_from` is a parameter and not an optimization: it is the thing that
decides whether this mechanism helps or hurts. The demo's `_owned` flag exists because it hit
the same wall from the other side — it clones **once** per stage per session and writes in
place thereafter, which v0.32 does not do (see §7).

---

## 5. Threading: read `CookResult.cooked_roi`, never the trace

A stage can **decline** the window — a gather, an unbounded reach, a refused rect — and hand
back a whole frame. `cooked_roi` is how the engine says so, and pasting a full-frame tensor
into a `w×h` slice is a crash, which is exactly what the demo did before it checked.

The engine records that on `tier_trace`, which is **thread-local** (`_local =
threading.local()`; `last_roi()` returns `(None, None)` off-thread). `run()` reads it on the
cook thread and stamps `CookResult.cooked_roi`, so the *result object* is safe to carry across
threads and the *trace* is not. CACHE-9 therefore reads `CookResult.cooked_roi` only — which
matters because SCHED-4 makes background cooks a real thing, and a region cook on the worker
whose window is read on the main thread would silently get `None` and paste a whole frame into
a window.

---

## 6. The ship gate (ROI-4), and what is measured

Every ROI feature ships behind the ROI-4 oracle lane; CACHE-9 is the one the risk register
already names as the correctness-sensitive item of the set, because it composes three
invariants at once (footprint over-approximation, ENG-12 immutability, lineage keys).

- **The differential row:** a patch-assembled frame equals the whole-frame cook. Over a chain
  with halo stages, on both devices, edited at several positions, asserted `< 1e-5` (the FUS-3
  convention — pointwise/morphology land bit-exact, conv within ~1 ulp of size-dependent
  kernel dispatch).
- **The never-sever rows:** a non-executable stage forces a whole-frame window at itself *and
  at every stage above it*; a declined window (`cooked_roi is None`) replaces rather than
  patches; a gather anywhere in the suffix widens to the frame.
- **The ownership row:** the cached base frame is byte-identical after a patch — i.e. the
  patch never reached the master.
- **The provenance row:** the patched frame's key differs from the base's, and carries it.

Measured, against the v0.31 baseline, on the shape PM-8 names (§2): `benchmarks/
region_recook_bench.py`.

**PM-8, measured (50 nodes, 2048², 512² window, sm_75).** A mid-graph param edit recooks in
**156 ms CPU (5.98×)** / **17.4 ms CUDA (9.88×)** against the 934 / 172 ms whole-graph recook.
The second half — "RAM stays under the governor budget" — needs one honest qualification: the
CACHE-5 governor is **host-driven**. `arbitrate()` is a call a host makes at its own safe
points, never a background thread (invariant #7 keeps everything off the default cook path).
Undriven, the comp's frame cache holds **1984 MB**; one `arbitrate()` leaves **960 MB** (CPU,
against a 1024 MB budget) / **704 MB** (CUDA, against 744 MB). Both numbers are reported,
because quoting only the second would claim a guarantee the engine does not make, and quoting
only the first would call a host's omission an engine failure.

*(Corrected in v0.33. This paragraph previously read "frees 1920 MB, leaving 64 MB" — which was
the governor being LIED TO: `_remove` forgot the per-device total, so `governed_bytes` reported
16000 MB for 1984 MB held and the governor evicted to the one-entry floor. That is a 16x
over-eviction wearing the costume of a governor landing exactly on budget. The accounting fix
and the corrected landing are in the v0.32 CHANGELOG.)*

**4K is not measured here, and the reason is arithmetic, not laziness.** This comp keeps one
full canvas per stage; at 4096² that is 268 MB × 50 = **13.4 GB**, which fits neither the 8 GB
card nor a sane CPU ceiling. The bench picks the largest square that fits and prints the
figure, so a row can never quietly claim a resolution it did not run.

---

## 7. Deferred, with the gate that would reopen each

- **Region recook of a FUSED region.** *Gate:* the reach analysis seeing through fusion's
  local variables (`_has_ungrounded_halo` is what blocks it) — a ROI/LANG item, not a caching
  one. Until then a fused chain uses CACHE-7's taps and recooks its window whole.
- **In-place patching of a cache-resident canvas.** v0.32 copies on every patch. The demo
  clones once and owns thereafter, which is strictly cheaper for a scrub. *Gate:* an
  engine-owned canvas type with an explicit ownership flag — which is really CACHE-8's
  residency work, since "who owns this buffer and where does it live" is one question.
- **fp16 regions.** ROI declines the window at fp16 with a measured reason. Same resolved-
  precision question CACHE-7 answers for taps; *gate:* the ROI-4 lane green at fp16.
- **Per-binding windows.** `RoiPlan.halo` is one scalar for the whole program, and
  `Footprint`'s four extents collapse to a square in every narrowable case. A per-edge window
  would tighten the composition. *Gate:* a measured case where the square costs more than the
  bookkeeping saves.

---

## 8. Canvas validity — the contract a multi-position edit needs (v0.32 audit)

A region cook leaves the canvases it patched correct **only over their composed windows**. That
is fine for a host that edits at one position. It is not fine the moment the user moves:

> Edit at `dirty_from=0` over a window in one corner. Now edit at `dirty_from=2` over the
> opposite corner. Stage 2 reads canvas 1 — which the first edit made correct only near the
> *first* corner. Everywhere else it still holds pre-edit pixels. **Measured: 2.17e-01** through
> documented `dirty_from` usage.

`chain_windows(halos, roi, dirty_from, valid=...)` takes the region each canvas is currently
correct over (`None` = the whole frame, which is what a whole cook leaves) and **returns `None`
when the plan cannot be served incrementally**. The host must then cook the whole chain from the
source.

**What does not work, recorded because it was the first fix attempted.** Widening the returned
windows to the full frame is useless. The upstream canvas is not being read too narrowly — it is
*wrong* outside the earlier patch, and no window choice at a downstream stage can repair a stale
input. Only re-cooking from far enough upstream fixes it, and `chain_windows` does not own
`dirty_from`, so it reports rather than pretends.

**How that was caught, which is the more useful lesson.** The first test asserted the *window
arithmetic* changed, and it passed — while the pixels were still wrong by 2.17e-01. The row that
found it cooks the chain twice and compares against a whole-frame reference, and it carries a
**negative control**: the same sequence without `valid` must still reproduce the stale ring, or
the guarded assertion proves nothing. Both halves are in `tests/test_v032_region.py`.
