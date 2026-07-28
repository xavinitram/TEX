"""tex_packing.py — how a cached frame is REPRESENTED in a cache tier (PREC-1, v0.33).

A cooked frame is fp32 because the *compute* contract says so (the whole-pipeline fp16/bf16
rejection stands, unqualified and untouched). Nothing in that contract says the frame must be
fp32 while it *sits in a cache waiting to be looked at again*. This module owns that second
question, and only that one.

THE SPLIT THIS FILE EXISTS TO KEEP CLEAN:

    compute dtype   what the interpreter/codegen tiers run at        — NOT here (tex_engine)
    wire dtype      what a cook returns to a host                    — NOT here (tex_marshalling)
    storage dtype   what a cached copy of a frame is kept as         — HERE, and nowhere else

The third is invisible to every program: a frame goes in as fp32 and comes out as fp32.
Between those two moments it may occupy fewer bytes.

The *conversion* is not ours either. `tex_io` has owned fp32↔storage since DATA-2, for files;
a cache tier is a different consumer of the same codec, so `pack`/`unpack` delegate to it. That
is not tidiness: the two implementations had already diverged on **349 of 262144 uint16 codes**
(half-away-from-zero here against `tex_io`'s round-half-to-even) one release after the second
was written. What this module owns is the POLICY — may this frame be reduced, and to what.

WHY THIS IS SAFE. The relevant comparison is not "is fp16 lossless" (it is not) but "is the
loss smaller than a difference this engine already ships". Three things decide it, and the
first is the only one that is scale-free:

  * RELATIVE ERROR, which is what a float actually bounds. Round-to-nearest costs at most half
    an ULP, so storage costs 2^-11 = 4.9e-4 for fp16 (10 mantissa bits) and 2^-8 = 3.9e-3 for
    bf16 (7 bits) — at every magnitude, on every corpus. The 8x between them is the entire
    difference and it cannot be argued away with a different test image.

  * THE 8-BIT DISPLAY QUANTUM, 1/255 = 3.9e-3, invariant #10's bar for an ACCEPTED program.
    Half an ULP in binade [2^k, 2^k+1) is 2^(k-11) for fp16 and 2^(k-8) for bf16, so
        bf16 crosses the quantum at values >= 2.0;  fp16 not until >= 16.0.
    On display-referred data both are under the bar — which is exactly why the naive test is
    misleading. A compositor's working space is SCENE-LINEAR and routinely exceeds 1.0, and
    that is the regime the recorded rejection was made in ("bf16 err > the 8-bit quantum",
    DEVELOPMENT.md; the interpreter records the measured 7.3e-3). bf16 stays a live negative
    control in the benchmark and the tests rather than a footnote — and it is not a
    representation this module will hand out, however it is asked.

  * INVARIANT #9's CPU-vs-GPU envelope, up to 6.1e-2. The same program cooked on the two
    devices this engine supports already differs by up to 125x more than fp16 storage does —
    and that is shipped, tested, and considered correct.

Measured (`benchmarks/storage_precision_bench.py`, 1024², 7 real compositing outputs, CPU and
CUDA — identical on both):

    codec    LDR maxdiff   scene-linear (x8)   8-bit codes changed   verdict
    fp16       2.4e-4           2.0e-3             0.25% - 2.0%      under the quantum in both
    bf16       2.0e-3           1.6e-2            13.3% - 18.6%      over it above 2.0

WHAT IS NOT DECIDED HERE, and must not be:

  * WHETHER a frame is eligible. That is the caller's `quality` tag. `None` — every caller
    that exists before this file did — means "store exactly what I cooked", byte for byte.
    Reduced storage is opt-in, per put, by a host that knows it is holding an interaction-tier
    preview. There is no auto-detection and no global switch (S-5: never silently retune a box).
  * COLOR vs DATA planes — and this is where an earlier draft of this file gave up too early.
    It said the engine "cannot tell them apart" because DATA-1's vocabulary
    (colorspace/premult/frame/extra) has no role field and named planes are DATA-6 (v0.35).
    Both facts are true and the conclusion was wrong: TEX has classified output KIND at the
    marshalling seam since M-3 — `map_inferred_type` turns an inferred TEXType into
    IMAGE / MASK / LATENT / INT / STRING. That is not a plane role, but it is exactly the
    distinction the split needs at this scale:

        IMAGE   colour data, display-referred or scene-linear   -> eligible
        MASK    a coverage/alpha channel; thresholds ride on it -> never packed
        LATENT  sampler input; not colour in any sense (M-3 already forces it fp32)  -> never

    So `choose_storage` takes `kind=`, and the caller that knows passes it. It stays a PURE
    function of its arguments — the cache never sniffs a tensor to guess a role, which is the
    S-5 line. When DATA-6 lands, a real plane role supersedes `kind` and the signature does not
    change.

    `storage="fp32"` remains as the explicit host pin, and one honest caveat about it: it is
    doing double duty as a *role assertion* rather than a codec choice. A caller passing it to
    mean "this plane is data" should expect to be SUPERSEDED by the role arm, not preserved by
    it — it stands in for metadata the engine does not have yet, and is not permanent API.
"""
from __future__ import annotations

# Module-level, unlike `tex_results`'s function-local torch imports. That convention exists to
# keep torch off the import path of a module the node loads; this module is itself imported
# lazily, from inside `ResultCache.put`/`get`, so by the time it exists torch is long loaded
# and the local imports bought nothing except three places to forget one of them.
import torch

from .tex_io import BufferDesc, decode_to_fp32, encode_from_fp32, _STORAGE_TORCH

# The quality tags `lineage_key(quality=...)` has carried since CACHE-1 shipped, spelled once
# so a host and the cache cannot disagree about the string. A preview frame and a final frame
# of the same program are DIFFERENT KEYS — which is the whole reason a lossy storage tier is
# admissible at all: nothing can serve a preview-quality frame to a caller that asked for the
# final one. The separation is the key's, not this module's.
PREVIEW = "preview"
FINAL = "final"

#: The reduced representations a cache tier may hold, named in `tex_io.STORAGE_DTYPES`'s
#: vocabulary rather than a second one — a host should not have to know `"fp32"` for a cache
#: and `"float32"` for a file. `"fp32"`/`torch.float32` remain accepted spellings of the PIN.
FP16 = "float16"
UINT16 = "uint16"
REDUCED = (FP16, UINT16)

#: Largest finite fp16. A frame carrying anything above this cannot be stored half — the value
#: would become `inf`, which is not "a bit less precise", it is a different number. Checked
#: rather than assumed, because a compositor's working space is scene-linear and unbounded
#: above: `hdr_glow` in the benchmark corpus already reaches 4.0, and a light source or a
#: divide can reach anything.
FP16_MAX = 65504.0

#: The 8-bit display quantum, and the code a viewer would show. This is the DECISION METRIC of
#: the whole item — the number `benchmarks/storage_precision_bench.py` and
#: `tests/test_v033_precision.py` are supposed to agree on — so it lives here rather than being
#: retyped correctly in both.
Q8_QUANTUM = 1.0 / 255.0

#: How a caller may spell each representation. Anything not in here is refused rather than
#: passed through: this module ships exactly two reduced representations, and an unvalidated
#: dtype pass-through would let `torch.bfloat16` — the codec both design notes conclude must
#: NOT ship — through the one door that was supposed to be argued.
_SPELLINGS = {
    FP16: FP16, torch.float16: FP16, "fp16": FP16,
    UINT16: UINT16, torch.uint16: UINT16,
}
_PINS = ("fp32", "float32", torch.float32)

#: Output KINDS eligible for reduced storage, in `tex_marshalling.map_inferred_type`'s
#: vocabulary. IMAGE is colour. MASK carries coverage that thresholds ride on, and LATENT is
#: sampler input that M-3 already forces to fp32 — neither is colour, and the roadmap's
#: "data planes fp32" is exactly them. `None` (a caller that does not know) is treated as
#: eligible, because `quality=PREVIEW` is itself an opt-in and the range gate still applies;
#: a caller that knows it is holding data says so.
COLOR_KINDS = ("IMAGE",)
DATA_KINDS = ("MASK", "LATENT", "INT", "STRING", "ARRAY")


def propagate_quality(own=None, upstream_qualities=()):
    """The quality tag a cook must be keyed under, given its own request and its INPUTS' tags.

    PREVIEW IS VIRAL, and this is the rule that makes the tier safe to compose. A cook that
    reads a preview frame produces preview bytes — fp16 storage quantization at ~4.9e-4 is an
    input error like any other, and the downstream program amplifies it exactly as
    `precision_policy`'s condition-number gate says it will. If that result were then stored
    under a `quality=None` key, a FINAL-quality lookup would be answered with preview-derived
    pixels: the final-render contract broken not by storing the wrong thing, but by LABELLING
    the wrong thing.

    So: any preview upstream forces preview. There is no "mostly final" tag, deliberately —
    a partial answer here is the laundering.

        propagate_quality(None, [FINAL, PREVIEW])  -> PREVIEW
        propagate_quality(FINAL, [PREVIEW])        -> PREVIEW      (a request cannot launder)
        propagate_quality(PREVIEW, [])             -> PREVIEW
        propagate_quality(None, [FINAL])           -> None         (unchanged)

    HONEST LIMIT, and it is the same shape as `boundary_lineage_key`'s: TEX sees `upstream` as
    opaque key strings and cannot recover a tag from a SHA. So the host that threads upstream
    keys must thread their qualities alongside — this function is the rule, not its enforcement.
    Under ComfyUI there is no TEX-internal upstream edge at all, so nothing to propagate;
    GRAPH-1's version counters are what let the engine carry this itself.
    """
    # A5/PROBE-10: a BARE STRING is one tag, not an iterable of characters. `tuple("preview")`
    # is `('p','r','e',...)`, which contains no PREVIEW — so the single-upstream spelling every
    # caller reaches for first returned the UNSAFE answer while the list spelling returned the
    # safe one. A rule whose safety depends on the caller's bracket choice is not a rule.
    if isinstance(upstream_qualities, str):
        upstream_qualities = (upstream_qualities,)
    return PREVIEW if (own == PREVIEW or PREVIEW in tuple(upstream_qualities)) else own


def q8(t):
    """The 8-bit code a viewer would display, as int16. The unit fidelity is judged in."""
    return (t.clamp(0.0, 1.0) / Q8_QUANTUM).round().to(torch.int16)


def choose_storage(t, *, quality=None, storage=None, kind=None):
    """The storage representation `t` should be kept as, or None for "store as cooked".

    Returns one of `REDUCED` (a `tex_io.STORAGE_DTYPES` name) or None.

    `quality`  the caller's tier tag — `PREVIEW` opts in, anything else (including None,
               the default every pre-v0.33 caller has) opts out.
    `storage`  WHICH reduced representation, not whether: `"uint16"` for fixed point,
               `"float16"` for half, `"fp32"` to PIN a frame at full precision.
    `kind`     the output KIND from `tex_marshalling.map_inferred_type` — IMAGE / MASK /
               LATENT / ... This is the colour-vs-data split, at the seam that already knows
               it. `None` means the caller did not say, which stays eligible: `PREVIEW` is
               opt-in and the range gate still applies.

    PURE in its arguments — no globals, no device query, no sniffing the tensor to guess a
    role. That is what makes a tier reproducible across a restart and reportable in a bug
    report, and it is S-5 applied to storage.

    Declines on every doubt. Over-declining costs bytes; over-accepting costs pixels, and only
    one of those is recoverable.
    """
    if not isinstance(t, torch.Tensor):
        return None
    if storage in _PINS:
        return None                           # an explicit pin: store as cooked, at any tier
    if kind is not None and kind not in COLOR_KINDS:
        return None                           # MASK / LATENT / a scalar wire: data, never packed
    # THE GATE, and it is the tier tag, not the storage hint. `storage=` selects WHICH reduced
    # representation; it never grants permission to use one. Without this ordering a caller
    # could reduce a FINAL frame by naming a codec, which is precisely the contract the whole
    # item promised not to touch.
    if quality != PREVIEW:
        return None                           # the default path, byte-identical to pre-v0.33
    if t.dtype is not torch.float32:
        # A8: fp32 SOURCES ONLY. The old test was "floating point and wider than 2 bytes",
        # which admits fp64 — and then the spill record spells `orig` through a table holding
        # only tex_io's four storage dtypes, so `float64` maps to `None` and the restored entry
        # FORGETS it was packed: float64 served before eviction, float16 after. Narrowing the
        # source is the fix that fails toward exactness, and fp32 is the only dtype a TEX cook
        # actually produces (invariant #4 forces coordinates fp32; outputs upcast on egress).
        # A uint8 mask and an already-half frame decline here too, as before: no win.
        return None
    want = FP16 if storage is None else _SPELLINGS.get(storage)
    if want is None:
        return None                           # an unsupported spelling: refuse, never guess
    # ONE feasibility gate, consulted by EVERY arm. It used to hang off the inferred arm only,
    # so an explicit `storage="float16"` pin skipped the range check and could store `inf`
    # — the single failure mode this module calls wrong rather than imprecise.
    return want if _admissible(t, want) else None


def _admissible(t, want) -> bool:
    """Can `want` actually hold `t`'s values? Exact, not heuristic — which is what lets this
    gate be trusted rather than tuned. One full-frame reduction, against a `put` that already
    costs ~1.1 ms at 512²."""
    lo, hi = _range(t)
    if lo is None:
        return False                          # an empty or un-reducible frame: store as cooked
    if want == UINT16:
        # Fixed point is only meaningful over the interval it maps. A frame that leaves [0,1]
        # would be CLIPPED — a silent 3.0-magnitude error on an HDR glow, measured — so an
        # out-of-range frame declines to full storage rather than being quietly clamped.
        return lo >= 0.0 and hi <= 1.0
    return max(abs(lo), abs(hi)) <= FP16_MAX


def _range(t):
    """`(min, max)` as floats in ONE reduction pass, or `(None, None)` if it cannot be taken
    (an empty frame, an exotic dtype). `aminmax` rather than a `.min()` and a `.max()`: this
    runs on every preview `put`, and one pass over a 4K frame is already 2.5 ms of memory
    traffic — paying it twice to learn two numbers from the same scan would be a choice."""
    try:
        lo, hi = torch.aminmax(t.detach())
        return float(lo), float(hi)
    except Exception:
        return None, None


def pack(t, want):
    """Convert a cooked frame to its stored representation. `want=None` is the identity.

    Delegates to `tex_io`'s codec — the one that has owned fp32↔storage since DATA-2. A second
    implementation is not free: the previous one here rounded half-away-from-zero where
    `tex_io` rounds half-to-even, and the two disagreed on 349 of 262144 uint16 codes."""
    if want is None:
        return t
    # Accept either spelling. `choose_storage` returns a name, but a caller holding a torch
    # dtype (a benchmark enumerating candidates, say) should not have to translate — and the
    # translation table already exists, so refusing would be strictness that buys nothing and
    # costs a `ValueError` deep inside `BufferDesc`.
    return encode_from_fp32(t, BufferDesc(storage=_SPELLINGS.get(want, want)))


def unpack(t, orig_dtype):
    """Restore a stored frame to the dtype it was cooked at. `orig_dtype=None` is the identity.

    This is a COPY, and that is why the upcast is free where it is felt: `ResultCache.get`
    already owes its caller an owned copy (copy-on-read is the actual ownership guarantee —
    freezing is a tripwire, not a fence), and `.to(fp32)` IS that copy. Measured at 1024²:
    upcast 0.04 ms CPU / 0.09 ms CUDA against a clone of 0.03 / 0.10. It replaces the clone
    rather than adding to it.
    """
    if orig_dtype is None:
        return t
    name = _NAME_OF.get(t.dtype)
    if name is None:
        return t if t.dtype == orig_dtype else t.to(orig_dtype)
    return decode_to_fp32(t, BufferDesc(storage=name)).to(orig_dtype)


#: torch dtype -> `tex_io.STORAGE_DTYPES` name, so `unpack` can name the representation a
#: stored tensor is already in. Derived from tex_io's own table rather than restated.
_NAME_OF = {dt: name for name, dt in _STORAGE_TORCH.items()}
