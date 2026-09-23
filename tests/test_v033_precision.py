"""PREC-1 (v0.33) — preview-tier storage precision.

The deferred decision, argued and then PINNED. What these rows defend, in the order that
matters if one of them ever goes red:

  1. THE DEFAULT PATH DID NOT MOVE. A `put` without a quality tag stores the cooked dtype,
     byte for byte. Every caller written before v0.33 is such a caller.
  2. THE FINAL-RENDER CONTRACT IS INTACT. A preview frame and a final frame are different
     keys, so no lookup can be answered by the wrong tier — and a final-tier `put` is never
     reduced, whatever the host asks for.
  3. THE LOSS IS THE ONE THAT WAS ARGUED. fp16 storage stays under the 8-bit display quantum;
     bf16 (the negative control) does not, which is why it is not offered.
  4. THE TIER IS INVISIBLE. A frame goes in fp32 and comes out fp32 — through RAM, and
     through the disk spill tier, which is where a dtype leak would otherwise hide.
"""
import tempfile

import torch

from helpers import make_gradient_frame as _frame
from TEX_Wrangle import tex_packing, tex_results
from TEX_Wrangle.tex_packing import q8 as _q8


def _cache(tmp, **kw):
    return tex_results.ResultCache(cache_dir=str(tmp), **kw)


# ── 1. the default path ───────────────────────────────────────────────────────

def test_v033_prec1_default_put_is_unchanged(r, tmp_path=None):
    """No quality tag = store exactly what was cooked. This is the whole of invariant #7's
    surface for this item: the ComfyUI node has no reference to ResultCache, and every
    engine-side caller that predates v0.33 passes no tag."""
    with tempfile.TemporaryDirectory() as d:
        c = _cache(d)
        f = _frame()
        c.put("k", f)
        got = c.get("k")
        entry = c._ram["k"]
        ok = (entry.tensor.dtype is torch.float32 and entry.orig_dtype is None
              and got.dtype is torch.float32 and torch.equal(got, f)
              and entry.nbytes == f.numel() * 4)
        r.ok("PREC-1: an untagged put stores the cooked dtype, bit-exact") if ok else \
            r.fail("PREC-1 default", f"dtype={entry.tensor.dtype} orig={entry.orig_dtype} bytes={entry.nbytes}")


def test_v033_prec1_final_tier_is_never_reduced(r):
    """The final-render contract is 'untouchable either way'. Asking for the final tier — or
    for nothing — must store full precision even if the host ALSO waved a storage hint at it."""
    with tempfile.TemporaryDirectory() as d:
        c = _cache(d)
        f = _frame()
        c.put("final", f, quality=tex_packing.FINAL)
        c.put("pinned", f, quality=tex_packing.PREVIEW, storage="fp32")
        ok = (c._ram["final"].tensor.dtype is torch.float32 and c._ram["final"].orig_dtype is None
              and c._ram["pinned"].tensor.dtype is torch.float32 and c._ram["pinned"].orig_dtype is None)
        r.ok("PREC-1: the final tier and an explicit fp32 pin are never reduced") if ok else \
            r.fail("PREC-1 final", f"final={c._ram['final'].tensor.dtype} "
                                   f"pinned={c._ram['pinned'].tensor.dtype}")


def test_v033_prec1_preview_and_final_are_different_keys(r):
    """The reason a lossy tier is admissible at all: nothing can serve a preview frame to a
    caller that asked for the final one, because they are not the same lookup. This is
    CACHE-1's `quality` component doing the load-bearing work, not a cache-side check."""
    base = dict(program_fp="p", device="cpu", precision="fp32")
    kp = tex_results.lineage_key(**base, quality=tex_packing.PREVIEW)
    kf = tex_results.lineage_key(**base, quality=tex_packing.FINAL)
    kn = tex_results.lineage_key(**base)
    ok = len({kp, kf, kn}) == 3
    r.ok("PREC-1: preview / final / untagged are three distinct lineage keys") if ok else \
        r.fail("PREC-1 keys", f"preview={kp[:8]} final={kf[:8]} none={kn[:8]}")


# ── 2. the storage tier itself ────────────────────────────────────────────────

def test_v033_prec1_preview_halves_the_bytes(r):
    """The capacity claim, measured through the cache's own accounting rather than asserted:
    `governed_bytes` is what the CACHE-5 governor reads, so this is the number that decides
    how many frames a budget holds."""
    with tempfile.TemporaryDirectory() as d:
        c = _cache(d)
        f = _frame(res=128)
        c.put("full", f)
        full = c.governed_bytes("cpu")
        c.clear()
        c.put("half", f, quality=tex_packing.PREVIEW)
        half = c.governed_bytes("cpu")
        ok = (c._ram["half"].tensor.dtype is torch.float16
              and c._ram["half"].orig_dtype is torch.float32
              and half * 2 == full)
        r.ok(f"PREC-1: a preview frame is stored half — {full} -> {half} bytes (2.00x)") if ok \
            else r.fail("PREC-1 bytes", f"full={full} half={half} dtype={c._ram['half'].tensor.dtype}")


def test_v033_prec1_storage_is_invisible_through_get(r):
    """A frame goes in fp32 and comes out fp32. If this ever fails, every consumer that reads
    `.dtype` off a cache hit is looking at a different tensor than it cooked."""
    with tempfile.TemporaryDirectory() as d:
        c = _cache(d)
        f = _frame()
        c.put("k", f, quality=tex_packing.PREVIEW)
        got, ro = c.get("k"), c.get("k", copy=False)
        ok = (got.dtype is torch.float32 and ro.dtype is torch.float32
              and got.shape == f.shape and not got.is_inference())
        r.ok("PREC-1: a half-stored frame is served fp32 (copy=True and copy=False)") if ok else \
            r.fail("PREC-1 invisible", f"get={got.dtype} copy=False={ro.dtype}")


def test_v033_prec1_relative_error_is_the_mantissa_bound(r):
    """THE argument, in the unit that is actually magnitude-independent.

    A float's storage error is RELATIVE: round-to-nearest costs at most half an ULP, so fp16
    (10 mantissa bits) costs 2^-11 and bf16 (7 bits) costs 2^-8, everywhere, at every scale.
    That 8x is the entire difference between the two, and it does not depend on which corpus
    anyone measures — which is why it, not an absolute number off one frame, is what is pinned
    here. The absolute view is the row below."""
    with tempfile.TemporaryDirectory() as d:
        c = _cache(d)
        f = _frame(res=256, scale=6.0)             # scene-linear: spans four binades
        c.put("k", f, quality=tex_packing.PREVIEW)
        got = c.get("k")
        nz = f.abs() > 0
        rel16 = float(((got - f).abs()[nz] / f.abs()[nz]).max())
        bf = f.to(torch.bfloat16).to(torch.float32)
        rel_bf = float(((bf - f).abs()[nz] / f.abs()[nz]).max())
        ok = rel16 <= 2.0 ** -11 and rel_bf > 2.0 ** -11 and rel_bf / max(rel16, 1e-12) > 4
        r.ok(f"PREC-1: fp16 storage costs {rel16:.2e} relative (bound 2^-11 = "
             f"{2.0**-11:.2e}); bf16 costs {rel_bf:.2e} — {rel_bf/max(rel16,1e-12):.0f}x more") \
            if ok else r.fail("PREC-1 relative", f"fp16={rel16:.3e} bf16={rel_bf:.3e}")


def test_v033_prec1_absolute_error_vs_the_8bit_quantum(r):
    """The absolute view, and the negative control that makes it mean something.

    Half an ULP in binade [2^k, 2^k+1) is 2^(k-11) for fp16 and 2^(k-8) for bf16, so against
    the 3.9e-3 = 2^-8 display quantum invariant #10 uses as its bar:

        bf16 crosses the quantum at values >= 2.0        fp16 not until >= 16.0

    A compositor's working space is scene-linear and routinely exceeds 1.0, so this is not a
    corner case — it is the ordinary case, and it is exactly why the recorded rejection named
    bf16. On DISPLAY-REFERRED data ([0,1]) both are under the bar; a test that measured only
    that would pass while proving nothing, so both regimes are asserted here."""
    QUANTUM = 3.9e-3
    with tempfile.TemporaryDirectory() as d:
        c = _cache(d)
        ldr, hdr = _frame(res=256), _frame(res=256, scale=8.0)
        c.put("ldr", ldr, quality=tex_packing.PREVIEW)
        c.put("hdr", hdr, quality=tex_packing.PREVIEW)
        e16_ldr = float((c.get("ldr") - ldr).abs().max())
        e16_hdr = float((c.get("hdr") - hdr).abs().max())
        ebf_hdr = float((hdr.to(torch.bfloat16).to(torch.float32) - hdr).abs().max())
        flips = float((_q8(c.get("ldr")) != _q8(ldr)).float().mean())
        ok = (e16_ldr < QUANTUM and e16_hdr < QUANTUM and ebf_hdr > QUANTUM)
        r.ok(f"PREC-1: fp16 stays under the {QUANTUM:.1e} quantum at 1.0 ({e16_ldr:.2e}) and "
             f"at 8.0 ({e16_hdr:.2e}); bf16 does not ({ebf_hdr:.2e}). "
             f"{flips:.2%} of 8-bit codes move by 1") if ok else \
            r.fail("PREC-1 quantum",
                   f"fp16 ldr={e16_ldr:.3e} hdr={e16_hdr:.3e} bf16 hdr={ebf_hdr:.3e}")


def test_v033_prec1_declines_what_half_cannot_represent(r):
    """The gate that makes over-acceptance impossible rather than unlikely. A value above
    fp16's range does not degrade — it becomes `inf`, a different number. The check is exact
    (a max reduction), not a heuristic, and it fails toward fp32."""
    with tempfile.TemporaryDirectory() as d:
        c = _cache(d)
        hdr = _frame(res=32, scale=1e5)            # 100000 > FP16_MAX = 65504
        c.put("hdr", hdr, quality=tex_packing.PREVIEW)
        served = c.get("hdr")
        mask = torch.ones(1, 8, 8, dtype=torch.uint8)
        c.put("mask", mask, quality=tex_packing.PREVIEW)
        ok = (c._ram["hdr"].tensor.dtype is torch.float32 and c._ram["hdr"].orig_dtype is None
              and bool(torch.isfinite(served).all()) and torch.equal(served, hdr)
              and c._ram["mask"].tensor.dtype is torch.uint8 and c._ram["mask"].orig_dtype is None)
        r.ok("PREC-1: an out-of-range frame and a uint8 mask both decline to full storage") \
            if ok else r.fail("PREC-1 decline",
                              f"hdr={c._ram['hdr'].tensor.dtype} mask={c._ram['mask'].tensor.dtype}")


# ── 3. the tier boundaries a dtype leak would hide behind ─────────────────────

def test_v033_prec1_survives_the_disk_spill_tier(r):
    """The spill/restore round-trip is where a representation leak would be intermittent —
    visible only after an eviction, i.e. only under memory pressure, i.e. only in the field.
    A restored preview frame must still be served fp32, and must still be HALF on disk."""
    with tempfile.TemporaryDirectory() as d:
        f = _frame(res=64)
        c = _cache(d, budget_mb=0)                 # every insert evicts the previous one
        c.put("a", f, quality=tex_packing.PREVIEW)
        c.put("b", f)                              # forces "a" out to disk
        got = c.get("a")
        entry = c._ram.get("a")
        ok = (c.spills >= 1 and got is not None and got.dtype is torch.float32
              and entry is not None and entry.tensor.dtype is torch.float16
              and entry.orig_dtype is torch.float32
              and float((got - f).abs().max()) < 3.9e-3)
        r.ok("PREC-1: a preview frame round-trips the disk tier still half, still served fp32") \
            if ok else r.fail("PREC-1 spill",
                              f"spills={c.spills} got={None if got is None else got.dtype} "
                              f"entry={None if entry is None else (entry.tensor.dtype, entry.orig_dtype)}")


def test_v033_prec1_patch_region_inherits_the_base_tier(r):
    """SUPERSEDES the v0.33.0 row `..._does_not_inherit_the_tier`, which asserted exactly the
    opposite. Recording the reversal rather than quietly editing the assertion, because the
    old row was not wrong about its own argument — it was answering the wrong question.

    **It said:** copy-on-patch stores under a NEW key, the tier is a property of that key, and
    inheriting the base's representation would make a frame's storage precision depend on the
    history of the cache rather than on what the caller asked for.

    **Why that loses (v0.33.2 A4):** the base's tier is not cache *history*, it is the fidelity
    of bytes that are physically in the output. A patch composes base pixels with patch pixels,
    and the result cannot be more faithful than its base — so storing it full-fp32 under a
    final-shaped key produced a frame whose out-of-window bytes still carried fp16 quantization
    (measured **1.89e-03**) while its tag claimed final. The old rule kept the tag honest about
    the CALL and let it lie about the CONTENT; PREC-1's viral rule wants the reverse, and this
    is the one seam where the cache can see the upstream well enough to enforce it.

    What is unchanged: an EXPLICIT `quality=PREVIEW` still stores preview, and a final base
    patched with no tag still stores final (`propagate_quality` only ever ratchets downward)."""
    with tempfile.TemporaryDirectory() as d:
        c = _cache(d)
        f = _frame(res=32)
        c.put("base", f, quality=tex_packing.PREVIEW)
        c.put("final_base", f)
        patch = torch.zeros(1, 8, 8, 4)
        out = c.patch_region("plain", patch, (4, 4, 8, 8, 32, 32), base_key="base")
        out2 = c.patch_region("prev", patch, (4, 4, 8, 8, 32, 32), base_key="base",
                              quality=tex_packing.PREVIEW)
        out3 = c.patch_region("fin", patch, (4, 4, 8, 8, 32, 32), base_key="final_base")
        ok = (out is not None and out2 is not None and out3 is not None
              and c._ram["plain"].quality == tex_packing.PREVIEW
              and c._ram["plain"].orig_dtype is torch.float32
              and c._ram["prev"].orig_dtype is torch.float32
              and c._ram["fin"].quality is None and c._ram["fin"].orig_dtype is None)
        r.ok("PREC-1: patch_region inherits a PREVIEW base's tier; a final base stays final") \
            if ok else \
            r.fail("PREC-1 patch",
                   f"plain={getattr(c._ram.get('plain'), 'quality', 'missing')}/"
                   f"{getattr(c._ram.get('plain'), 'orig_dtype', 'missing')} "
                   f"prev={getattr(c._ram.get('prev'), 'orig_dtype', 'missing')} "
                   f"fin={getattr(c._ram.get('fin'), 'quality', 'missing')}/"
                   f"{getattr(c._ram.get('fin'), 'orig_dtype', 'missing')}")


def test_v033_prec1_is_absent_from_the_default_comfyui_path(r):
    """Invariant #7 as a source canary, the shape every cache feature since ROI-3 has shipped
    with. The default ComfyUI cook must not merely be unaffected in a benchmark — the module
    must be unreachable from it, which is a property a grep can decide and a timing cannot."""
    import pathlib
    node = (pathlib.Path(__file__).resolve().parent.parent / "tex_node.py").read_text(
        encoding="utf-8")
    hits = [n for n in ("tex_packing", "tex_results", "choose_storage", "PREVIEW") if n in node]
    r.ok("PREC-1: tex_node.py references neither the packing policy nor the frame cache") \
        if not hits else r.fail("PREC-1 invariant#7", f"tex_node.py mentions {hits}")


def test_v033_prec1_choose_storage_is_the_only_decision_point(r):
    """`choose_storage` is a pure function of (tensor, quality, storage) — no globals, no
    device query, no environment. That is what makes the tier reproducible across a restart
    and reportable in a bug report, and it is the S-5 'never silently retune a box' rule
    applied to storage."""
    f = _frame(res=16)
    cases = [
        (dict(quality=None), None),
        (dict(quality=tex_packing.FINAL), None),
        (dict(quality=tex_packing.PREVIEW), tex_packing.FP16),
        (dict(quality=tex_packing.PREVIEW, storage="fp32"), None),
        (dict(quality="PREVIEW"), None),            # tags are exact strings, not case-folded
        # TRK-90: a storage hint with NO quality tag must not reduce on its own — `storage=`
        # selects WHICH representation, it never grants permission (the gate is `quality`).
        (dict(quality=None, storage="fp16"), None),
        # mask_eligible is just another argument: pure and exact-match like the rest.
        (dict(quality=tex_packing.PREVIEW, kind="MASK", mask_eligible=True), tex_packing.FP16),
    ]
    bad = [(kw, tex_packing.choose_storage(f, **kw)) for kw, want in cases
           if tex_packing.choose_storage(f, **kw) is not want]
    stable = all(tex_packing.choose_storage(f, quality=tex_packing.PREVIEW) == tex_packing.FP16
                 for _ in range(5))
    stable_knob = all(
        tex_packing.choose_storage(f, quality=tex_packing.PREVIEW, kind="MASK",
                                    mask_eligible=True) == tex_packing.FP16
        for _ in range(5))
    r.ok("PREC-1: choose_storage is pure, exact-match, and stable across calls, "
         "including with mask_eligible") \
        if not bad and stable and stable_knob else \
        r.fail("PREC-1 choose", f"{bad} stable={stable} stable_knob={stable_knob}")


# ── §2.1 decisions 3 and 5 (doc 41) ───────────────────────────────────────────

def test_v033_prec1_colour_data_split_at_the_kind_seam(r):
    """Decision 3. An earlier draft declared this un-implementable because DATA-1 has no plane
    role. True, and beside the point: `tex_marshalling.map_inferred_type` has classified output
    KIND since M-3, and IMAGE-vs-MASK/LATENT is exactly the colour-vs-data line at this scale.

    `kind` stays an ARGUMENT — the cache never sniffs a tensor to guess a role (S-5)."""
    f = _frame(res=32)
    P = tex_packing.PREVIEW
    rows = {k: tex_packing.choose_storage(f, quality=P, kind=k)
            for k in ("IMAGE", "MASK", "LATENT", "INT", "STRING", None)}
    ok = (rows["IMAGE"] == tex_packing.FP16 and rows[None] == tex_packing.FP16
          and all(rows[k] is None for k in ("MASK", "LATENT", "INT", "STRING")))
    r.ok("PREC-1: IMAGE packs; MASK/LATENT/scalar wires never do; unknown stays eligible") \
        if ok else r.fail("PREC-1 kind split", f"{rows}")


def test_v033_prec1_kind_reaches_the_cache(r):
    """The seam is only real if `put` carries it."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        c = _cache(d)
        f = _frame(res=32)
        c.put("img", f, quality=tex_packing.PREVIEW, kind="IMAGE")
        c.put("msk", f, quality=tex_packing.PREVIEW, kind="MASK")
        ok = (c._ram["img"].orig_dtype is torch.float32
              and c._ram["msk"].orig_dtype is None)
        r.ok("PREC-1: put(kind=…) packs an IMAGE and refuses a MASK") if ok else \
            r.fail("PREC-1 kind put", f"img={c._ram['img'].orig_dtype} "
                                      f"msk={c._ram['msk'].orig_dtype}")


# ── mask_eligible: a per-put opt-in that widens the kind gate for MASK only ───

def test_v033_prec1_mask_eligible_knob_matrix(r):
    """The whole truth table for `mask_eligible`, on a genuine rank-3 `[1,32,32]` mask —
    ComfyUI's own MASK shape, with no channel dimension. The knob widens exactly ONE thing:
    `kind="MASK"` past the kind gate, and only when quality/storage/range already agree the
    frame may pack. Every other kind, and every other gate, must see it not at all."""
    P, F = tex_packing.PREVIEW, tex_packing.FINAL
    m = _frame(res=32, c=1).squeeze(-1)                     # [1,32,32], values in [0,1]
    hi = _frame(res=32, c=1, scale=7e4).squeeze(-1)         # [1,32,32], values in [0,7e4]
    cases = {
        "MASK+knob+PREVIEW packs":
            (dict(quality=P, kind="MASK", mask_eligible=True), tex_packing.FP16),
        "MASK+knob+FINAL stays refused":
            (dict(quality=F, kind="MASK", mask_eligible=True), None),
        "MASK+knob+untagged stays refused":
            (dict(quality=None, kind="MASK", mask_eligible=True), None),
        "MASK+knob+fp32 pin stays refused":
            (dict(quality=P, kind="MASK", mask_eligible=True, storage="fp32"), None),
        "LATENT+knob is still data":
            (dict(quality=P, kind="LATENT", mask_eligible=True), None),
        "INT+knob is still data":
            (dict(quality=P, kind="INT", mask_eligible=True), None),
        "STRING+knob is still data":
            (dict(quality=P, kind="STRING", mask_eligible=True), None),
        "ARRAY+knob is still data":
            (dict(quality=P, kind="ARRAY", mask_eligible=True), None),
        "IMAGE+knob is a no-op":
            (dict(quality=P, kind="IMAGE", mask_eligible=True), tex_packing.FP16),
        "unknown-kind+knob is a no-op":
            (dict(quality=P, kind=None, mask_eligible=True), tex_packing.FP16),
    }
    bad = {name: (tex_packing.choose_storage(m, **kw), want)
           for name, (kw, want) in cases.items()
           if tex_packing.choose_storage(m, **kw) is not want}
    uint16_ok = (tex_packing.choose_storage(m, quality=P, kind="MASK", mask_eligible=True,
                                             storage="uint16") == tex_packing.UINT16
                 and tex_packing.choose_storage(m * 2, quality=P, kind="MASK",
                                                 mask_eligible=True, storage="uint16") is None)
    hi_ok = tex_packing.choose_storage(hi, quality=P, kind="MASK", mask_eligible=True) is None
    r.ok("mask_eligible: the knob matrix matches the design's truth table exactly") \
        if not bad and uint16_ok and hi_ok else \
        r.fail("mask_eligible knob matrix", f"bad={bad} uint16_ok={uint16_ok} hi_ok={hi_ok}")


def test_v033_prec1_mask_eligible_packs_through_put(r):
    """The knob reaches `ResultCache.put` end to end: a MASK put with `mask_eligible=True`
    packs to fp16 and reads back within half an fp16 ULP on [0.5,1) — the same envelope
    PREC-1 already ships for IMAGE (K2)."""
    with tempfile.TemporaryDirectory() as d:
        c = _cache(d)
        m = _frame(res=32, c=1).squeeze(-1)                 # [1,32,32]
        c.put("m", m, quality=tex_packing.PREVIEW, kind="MASK", mask_eligible=True)
        entry = c._ram["m"]
        got = c.get("m")
        maxdiff = float((got - m).abs().max())
        ok = (entry.tensor.dtype is torch.float16 and entry.orig_dtype is torch.float32
              and got.dtype is torch.float32 and maxdiff <= 2.45e-4)
        r.ok(f"mask_eligible: an opted-in MASK put packs to fp16, reads back at "
             f"{maxdiff:.2e} (<= 2.45e-4)") if ok else \
            r.fail("mask_eligible put", f"dtype={entry.tensor.dtype} orig={entry.orig_dtype} "
                                        f"got={got.dtype} maxdiff={maxdiff:.3e}")


def test_v033_prec1_mask_eligible_guard_default_refuses(r):
    """Guard row (K3): with the knob absent — the call shape every caller before it existed
    uses — a MASK put stays float32 for a genuine rank-3 `[1,32,32]` mask AND the rank-4
    `[1,32,32,1]` shape `test_v033_prec1_kind_reaches_the_cache` already covers. The default
    must not move for either shape."""
    with tempfile.TemporaryDirectory() as d:
        c = _cache(d)
        m3 = _frame(res=32, c=1).squeeze(-1)                # [1,32,32]
        m4 = _frame(res=32, c=1)                            # [1,32,32,1]
        c.put("m3", m3, quality=tex_packing.PREVIEW, kind="MASK")
        c.put("m4", m4, quality=tex_packing.PREVIEW, kind="MASK")
        ok = (c._ram["m3"].tensor.dtype is torch.float32 and c._ram["m3"].orig_dtype is None
              and c._ram["m4"].tensor.dtype is torch.float32 and c._ram["m4"].orig_dtype is None)
        r.ok("mask_eligible: a MASK put with no knob stays float32, rank-3 and rank-4 alike") \
            if ok else r.fail("mask_eligible guard",
                               f"m3={c._ram['m3'].tensor.dtype}/{c._ram['m3'].orig_dtype} "
                               f"m4={c._ram['m4'].tensor.dtype}/{c._ram['m4'].orig_dtype}")


def test_v033_prec1_mask_eligible_patch_region_ratchet(r):
    """`patch_region` gained no knob of its own — it ratchets on the base's STORED
    representation (v0.33.2 H3), so a patch over a mask that came in through `mask_eligible`
    packs exactly like a patch over a packed IMAGE, and becomes discoverable by
    `preview_entries()` for the idle requalifier to reach (K4). The unopted-in MASK row this
    same ratchet defends (`test_v0332_h3_the_ratchet_never_packs_a_frame_put_refused_to_pack`)
    asserts the no-knob case, which nothing here moves."""
    with tempfile.TemporaryDirectory() as d:
        c = _cache(d)
        m = _frame(res=32, c=1).squeeze(-1)                 # [1,32,32]
        c.put("base", m, quality=tex_packing.PREVIEW, kind="MASK", mask_eligible=True)
        patch = torch.zeros(1, 8, 8)
        out = c.patch_region("mp", patch, (4, 4, 8, 8, 32, 32), base_key="base")
        entry = c._ram.get("mp")
        ok = (out is not None and entry is not None
              and entry.tensor.dtype is torch.float16 and entry.orig_dtype is torch.float32
              and entry.quality == tex_packing.PREVIEW
              and "mp" in c.preview_entries())
        r.ok("mask_eligible: patch_region over a knob-packed mask packs and is preview-listed") \
            if ok else r.fail("mask_eligible patch_region",
                               f"out={out is not None} entry={entry}")


def test_v033_prec1_mask_eligible_survives_the_disk_spill_tier(r):
    """The disk-spill round trip (the same pattern as `..._survives_the_disk_spill_tier`) is
    where a representation leak would be intermittent — visible only under memory pressure.
    A knob-packed mask must come back fp32, bit-identical to what it read before it spilled,
    exactly like a knob-packed IMAGE already does (K5)."""
    with tempfile.TemporaryDirectory() as d:
        m = _frame(res=32, c=1).squeeze(-1)                 # [1,32,32]
        c = _cache(d, budget_mb=0)                          # every insert evicts the previous
        c.put("a", m, quality=tex_packing.PREVIEW, kind="MASK", mask_eligible=True)
        pre_spill = c.get("a")
        c.put("b", m)                                       # forces "a" out to disk
        got = c.get("a")
        entry = c._ram.get("a")
        ok = (c.spills >= 1 and got is not None and got.dtype is torch.float32
              and entry is not None and entry.tensor.dtype is torch.float16
              and entry.orig_dtype is torch.float32
              and torch.equal(got, pre_spill))
        r.ok("mask_eligible: a knob-packed mask round-trips the disk tier, bit-identical") \
            if ok else r.fail("mask_eligible spill",
                               f"spills={c.spills} got={None if got is None else got.dtype} "
                               f"equal={got is not None and torch.equal(got, pre_spill)}")


def test_v033_prec1_preview_is_viral(r):
    """Decision 5, and the one that is a correctness hole rather than a policy choice: a cook
    reading a preview frame produces preview bytes. Storing THAT under a `quality=None` key
    would let a final-quality lookup be answered with preview-derived pixels — the final-render
    contract broken by mislabelling rather than by mis-storing.

    Any preview upstream forces preview; a request cannot launder it back."""
    P, F = tex_packing.PREVIEW, tex_packing.FINAL
    cases = {
        "clean stays clean": (tex_packing.propagate_quality(None, [F, None]), None),
        "one preview upstream infects": (tex_packing.propagate_quality(None, [F, P]), P),
        "a FINAL request cannot launder": (tex_packing.propagate_quality(F, [P]), P),
        "own preview with no upstream": (tex_packing.propagate_quality(P, []), P),
        "no inputs at all": (tex_packing.propagate_quality(None, []), None),
    }
    bad = {k: v for k, (v, want) in cases.items() if v != want}
    r.ok("PREC-1: preview is viral — a preview upstream forces a preview key") if not bad \
        else r.fail("PREC-1 viral quality", f"{bad}")
