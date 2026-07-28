# PREC-1 — preview-tier storage precision (the deferred decision)

*Design/decision note, v0.33.0. Companion: `tex_packing.py`, `tests/test_v033_precision.py`,
`benchmarks/storage_precision_bench.py`.*

## 0. What is being decided, and what is not

The register carries an **unqualified rejection**:

> Whole-pipeline fp16 & bf16 IMAGE — accuracy (bf16 err > the 8-bit quantum). *(DEVELOPMENT.md)*

and a matching **sanctioned reopening**:

> *Preview-tier fp16 storage* (PREC-1) — the recorded rejection is unqualified; any carve-out
> is argued fresh. *(docs/roadmap.md)*

"Argued fresh" is the whole job of this document. It does not narrow the rejection, appeal to
it, or lean on the fact that another compositor ships half working precision. It asks one
question that the rejection did not answer because it was not asked:

> A frame has already been cooked in fp32. It is now sitting in the interaction-tier cache,
> waiting to be looked at again. **Must it occupy 4 bytes per channel while it waits?**

Three dtypes, three different questions, and only the third is on the table:

| | what it is | decided by | this doc |
|---|---|---|---|
| compute dtype | what the interpreter/codegen tiers run at | `tex_engine` / `precision_policy` | untouched |
| wire dtype | what a cook returns to a host | `tex_marshalling` | untouched |
| **storage dtype** | what a *cached copy* of a finished frame is kept as | `tex_packing` | **decided here** |

The final-render contract is untouched in both directions: a final-quality frame is never
stored reduced, and a preview-quality frame can never be *served* to a caller that asked for
the final one — because they are not the same lookup (§3).

## 1. The measurement

`benchmarks/storage_precision_bench.py`, 1024², seven real compositing outputs, CPU and CUDA
(identical to the digit on both — the arithmetic is the same). `q8_flip` is the fraction of
pixels whose **8-bit display code changes**, which is the unit a preview frame is actually
consumed in.

| codec | bytes | LDR maxdiff | scene-linear (×8) maxdiff | q8_flip (LDR) | pack ms | unpack ms | clone ms |
|---|---|---|---|---|---|---|---|
| fp32 (today) | 4/ch | — | — | — | — | — | 0.03 |
| **fp16** | **2/ch** | **2.4e-4** | **2.0e-3** | **0.25 – 2.0 %** | 0.04 | 0.04 | 0.03 |
| bf16 | 2/ch | 2.0e-3 | 1.6e-2 | 13.3 – 18.6 % | 0.05 | 0.05 | 0.03 |
| uint16 | 2/ch | 7.7e-6 | **clips** (err 3.0 – 7.0) | 3e-6 | 0.50 | 0.20 | 0.03 |

Capacity is exact arithmetic, not a measurement: 2.00× frames held at any budget. At 4K a
frame goes 256 MB → 128 MB, so a 1 GB frame budget holds 8 instead of 4.

## 2. The argument

**a. Relative error is the only scale-free unit, and it is a mantissa count.** Round-to-nearest
costs at most half an ULP, so storage costs `2^-11 = 4.9e-4` for fp16 (10 mantissa bits) and
`2^-8 = 3.9e-3` for bf16 (7 bits) — at every magnitude, on every corpus. The 8× between them
is the entire difference between the two candidates and no choice of test image can move it.
This is what `test_v033_prec1_relative_error_is_the_mantissa_bound` pins.

**b. Against the 8-bit display quantum, the two land on opposite sides — but only in the
regime a compositor actually works in.** Half an ULP in binade `[2^k, 2^k+1)` is `2^(k-11)`
for fp16 and `2^(k-8)` for bf16, so against `1/255 = 3.9e-3 = 2^-8`:

> **bf16 crosses the quantum at values ≥ 2.0. fp16 does not cross it until ≥ 16.0.**

On display-referred data (`[0,1]`) *both* sit under the bar. A decision made on an LDR test
corpus would therefore have cleared bf16 — and the register says bf16 was measured at 7.3e-3,
i.e. over the bar. There is no contradiction: TEX's working space is **scene-linear**, and the
benchmark corpus reaches 4.0 and 8.0 on ordinary glow and exposure nodes. That is the regime
the rejection was made in, and it is the regime the tests assert in
(`test_v033_prec1_absolute_error_vs_the_8bit_quantum` checks fp16 at both 1.0 and 8.0, and
requires bf16 to fail at 8.0). **The first draft of that test measured only LDR and passed for
the wrong reason** — it is recorded here because the same mistake is the easy one to repeat.

**c. The loss is far inside a difference the engine already ships.** Invariant #9's CPU-vs-GPU
envelope is up to **6.1e-2**: the same program cooked on the two devices TEX supports already
differs by up to **125×** more than fp16 storage does, and that is tested, documented, and
considered correct. A cache tier whose error is two orders of magnitude below the difference
between two *correct* cooks of the same program is not the weak link.

**d. The visible consequence is ±1 code on ~1 % of pixels.** Not zero — the honest number.
fp16 storage moves 0.25 – 2.0 % of 8-bit display codes, always by exactly one step. bf16 moves
13 – 19 %.

**e. It is free where it is paid.** `get` already owes its caller an owned copy (copy-on-read
is the real ownership guarantee; freezing is a tripwire, not a fence), and `.to(fp32)` **is**
that copy. The upcast replaces the clone rather than adding to it: 0.04 ms vs 0.03 ms at 1024².
`put` pays one downcast (0.04 ms) plus one range reduction, against a `put` that already costs
~1.1 ms at 512².

### Verdict

**Ship fp16 as an opt-in storage tier for interaction-tier frames. Do not ship bf16.**
The rejection stands exactly as written — it is about compute and wire dtype, and bf16 fails
the storage test too, on its own merits, measured here rather than assumed.

## 3. What was built

**`tex_packing.py`** — the only place a storage representation is chosen.
`choose_storage(t, *, quality, storage)` is a pure function of its arguments: no globals, no
device query, no environment. That is what makes a tier reproducible across a restart and
reportable in a bug report (S-5: never silently retune a box).

```
quality=None (the default every pre-v0.33 caller uses)  -> store exactly as cooked
quality=FINAL                                           -> store exactly as cooked
quality=PREVIEW                                         -> fp16, if the frame allows it
quality=PREVIEW, storage="fp32"                         -> pinned full precision
```

**`ResultCache`** grew a sixth entry slot, `orig_dtype`, and `put` grew `quality=`/`storage=`.
`put` decides the representation; `_admit` (new, the shared insert body) applies it; `get`
upcasts through `orig_dtype`, so **storage precision is invisible from outside the class**.
`_spill` writes `orig_dtype` beside the frame and `_restore` re-admits through `_admit` rather
than `put` — re-running the policy on a restored half frame would ask about a tensor whose
cooked dtype no longer exists. A preview frame is therefore half on **disk** too, for free.

**Key separation is CACHE-1's, not the cache's.** `lineage_key(quality=...)` has carried the
component since v0.25 with no caller; PREC-1 is its first. Preview, final, and untagged are
three distinct keys, so a lossy tier cannot leak into a lookup that did not ask for it. The
cache performs no tier check at all, which is the right altitude: the identity question was
already solved.

## 4. The honest gaps

**The colour/data plane split is not implemented as stated.** The roadmap's shape is "colour
planes half / data planes fp32". TEX cannot tell them apart today: DATA-1's vocabulary is
`colorspace / premult / frame / extra` with **no role field**, and named planes are DATA-6
(v0.35). Pretending otherwise would mean inferring a role from pixels, which is exactly the
kind of silent auto-tuning S-5 forbids. So the split is expressed with the two instruments
that do exist, and both fail toward fp32:

* an explicit `storage="fp32"` pin, for a host that knows a binding is depth/normals/motion;
* a value-range gate — anything above `FP16_MAX` declines, because overflow to `inf` is not
  "less precise", it is a different number.

When DATA-6 lands, `choose_storage` grows a role arm and no caller changes.

**LATENT is not auto-excluded.** A latent is a data plane by any reading, but `ResultCache`
receives a tensor, not a wire type, and cannot see it. A host caching latents at preview
quality must pass `storage="fp32"`. Recorded as a gap rather than papered over with a shape
heuristic ([B,C,H,W] with C=4 is also an ordinary RGBA image).

**uint16 is measurably better inside `[0,1]` and is not shipped here.** 7.7e-6 versus fp16's
2.4e-4 — 32× — at the same width. It clips everything outside the unit interval, so it needs a
provable range, and it costs 12× more to pack. That trade belongs to **CACHE-8**, which owns
"half/uint16 packing" as its own item and inherits this module's seam.

**Nothing in the engine yet cooks at preview quality.** PREC-1 ships the mechanism, the key
separation, and the argument; deciding *when* to cook a preview and when to re-cook full
precision on idle is host policy, and the runway (doc 39 §4) names it as such.

## 5. Gates

Invariant #7 by construction and by canary: `tex_node.py` references neither `tex_packing` nor
`tex_results`, asserted as a source grep rather than a timing. The default `put` path stores
the cooked dtype bit-for-bit, which is the only behaviour any existing caller can observe.
