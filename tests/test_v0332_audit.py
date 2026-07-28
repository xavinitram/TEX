"""v0.33.2 — the v0.33.1 release-audit findings, each pinned by a row that fails pre-fix.

Same shape as v0.33.1 and for the same reason: every one of these lives behind an ARMED path
(the spill tier, `set_vram_budget`/GOV-1 residency, or an opt-in `quality=`/`storage=` tag),
so none is reachable from the default ComfyUI cook — and every one is closed anyway, because
"unreachable today" is a property of the wiring, not of the code.

  A1  two in-flight spills of the SAME key could commit in the wrong order: a `put` that
      replaced a frame with new pixels was overwritten on disk by the OLDER frame's write,
      and the disk tier then served stale pixels forever
  A2  a `get` racing `clear(disk=True)` RESURRECTED the cleared frame in RAM — `_spill` had
      the generation check for exactly this class, `_restore`'s re-admit did not
  A3  `clear(disk=True)`'s tail asserted a definite empty index over a frame legitimately
      spilled during its unlocked unlink walk (the P0-6 family, through `clear`'s door)
  A4  `patch_region` laundered a preview base into a final-shaped result — the one seam where
      the cache can SEE the upstream tag, so the viral rule is enforceable rather than merely
      documented. The tag is now a stored entry slot and a `.frame` field (fmt 1 -> 2).
  A5  closures: a single tag passed to `propagate_quality` as a bare string returned the
      UNSAFE answer; a future `.frame` format was decoded rather than refused; a shape
      mismatch RAISED out of a method whose contract is to refuse; and disarming residency
      cancelled neither its queued demotions nor one already in flight. (A fifth, a lock-depth
      early-out in `_promote`, was WITHDRAWN — the row here pins the invariant that restores.)
  H1-H5  found by an adversarial hunt over this tree AFTER A/B/C were complete and before the
      tag. H1 and H2 are A1 and A2 failing on interleavings their own pins could not reach:
      the spill ticket was claimed after `_drain_spills` released the lock, and a restore could
      outrun `clear(disk=True)`'s unlocked unlink walk. H3 is A4's ratchet taking a tag as
      licence to pack a frame `put` refuses to pack (a MASK). H4 is the P0-6 family through a
      third unlocked directory walk. H5 is a spill write whose failure was discarded.
"""
import ast
import inspect
import os
import pickle
import tempfile
import textwrap
import threading

import torch

from helpers import devices as _devices, make_gradient_frame as _frame
from TEX_Wrangle import tex_packing, tex_results


def _flat(v, res=64, ch=4):
    return torch.full((1, res, res, ch), float(v))


# ── A1 ────────────────────────────────────────────────────────────────────────

def test_v0332_a1_a_stale_spill_never_overwrites_the_winner(r):
    """`put(K, v1)` -> evicted -> its spill in flight; `put(K, v2)` (the replace path `_admit`'s
    docstring blesses) -> evicted -> its spill COMPLETES and indexes K. The first write then
    lands on top and the disk tier serves v1 — after both puts and both spills returned.

    The generation counter could not catch this (no `clear` ran, so the generation matches);
    ordering between two writes of the same key is a different fact from liveness. A per-key
    monotonic ticket claimed under `_lock` says who is newest, and a per-key write lock makes
    the check and the write one critical section — a check outside the write is not enough,
    because a stale writer that has already passed it can still overwrite the winner."""
    from TEX_Wrangle.tex_runtime import streams
    K = "k_target"
    with tempfile.TemporaryDirectory() as d:
        c = tex_results.ResultCache(budget_mb=0, cache_dir=d)   # 0: every put evicts all but MRU
        c.put(K, _flat(1.0))

        gate_in, resume = threading.Event(), threading.Event()
        real_egress, state = streams.egress, {"armed": True}

        def gated_egress(src, **kw):
            # Hold the OLD frame's spill between its ticket claim and its write. Identified by
            # its pixels so the filler spills pass straight through.
            if state["armed"] and float(src.reshape(-1)[0]) == 1.0:
                state["armed"] = False
                gate_in.set()
                assert resume.wait(20), "gate timeout"
            return real_egress(src, **kw)

        streams.egress = gated_egress
        t = threading.Thread(target=lambda: c.put("f1", _flat(0.0)))  # evicts K(v1) -> spill v1
        t.start()
        try:
            assert gate_in.wait(20), "the v1 spill never reached the gate"
            c.put(K, _flat(2.0))            # the blessed replace: new pixels under the same key
            c.put("f2", _flat(0.0))         # evicts K(v2) -> its spill writes and indexes K
            resume.set()
            t.join(20)
        finally:
            streams.egress = real_egress
        got = c.get(K)                      # RAM holds only f2 — this is a disk restore
        mean = None if got is None else float(got.float().mean())
        r.ok("A1: the disk tier serves the LAST write of a key, not the last writer to finish") \
            if mean == 2.0 else \
            r.fail("A1 spill inversion",
                   f"disk served {mean} after put(K, 2.0) and its spill both completed")


def test_v0332_a1_a_check_and_write_are_one_critical_section(r):
    """The functional row above cannot tell "checked, then wrote" from "checked AND wrote
    atomically" — both serve v2 on the interleaving it can force. This one reads the source:
    the ticket comparison and `_atomic_pickle` must sit inside the SAME `with` block.

    Pinned structurally because the first two attempts at A1 both failed here specifically. A
    ticket-only fix stopped the stale pixels but became lossy (the stale writer deleted the
    file it had just overwritten), and a pre-write check outside the lock still lost the race
    when the winner's write was the one that got delayed."""
    src = textwrap.dedent(inspect.getsource(tex_results.ResultCache._spill))
    fn = ast.parse(src).body[0]
    guarded = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.With):
            continue
        if not any(isinstance(i.context_expr, ast.Name) and i.context_expr.id == "wlock"
                   for i in node.items):
            continue
        body = list(ast.walk(node))
        wrote = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id == "_atomic_pickle" for n in body)
        checked = any(isinstance(n, ast.Attribute) and n.attr == "_spill_seq" for n in body)
        guarded.append(wrote and checked)
    writes = sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == "_atomic_pickle" for n in ast.walk(fn))
    ok = guarded and all(guarded) and writes == 1
    r.ok("A1: the ticket check and the frame write share one per-key critical section") if ok \
        else r.fail("A1 critical section",
                    f"per-key `with wlock` blocks holding both check and write: {guarded}, "
                    f"_atomic_pickle call sites in _spill: {writes}")


# ── A2 ────────────────────────────────────────────────────────────────────────

def test_v0332_a2_a_restore_cannot_resurrect_a_cleared_frame(r):
    """`_restore` runs its file read OUTSIDE the lock by design, so `clear(disk=True)` can take
    the lock inside that window — and the re-admit that follows put the cleared frame straight
    back into `_ram`, where gets issued AFTER `clear()` returned were served from it.

    Linearizability is not the question: the racing `get` may legitimately return the frame it
    was asked for before the clear. What must not survive is the INSERT. The check therefore
    belongs inside `_admit`'s locked section and nowhere earlier — a pre-`_admit` check reads a
    generation that can move before the insert takes the lock."""
    class WideWindow(tex_results.ResultCache):
        """Widens the natural window with an event. No logic is altered."""
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.trip = False
            self.in_window, self.resume = threading.Event(), threading.Event()

        def _admit(self, *a, **kw):
            if self.trip:
                self.trip = False                     # only the raced restore pauses
                self.in_window.set()
                self.resume.wait(20)
            return super()._admit(*a, **kw)

    with tempfile.TemporaryDirectory() as d:
        c = WideWindow(cache_dir=d, budget_mb=0)
        c.put("k", _frame(res=64))
        c.put("j", _frame(res=8))                     # evicts "k" to the disk tier
        assert "k" not in c._ram and c.spills == 1, "setup: k should have spilled"

        c.trip = True
        t = threading.Thread(target=lambda: c.get("k"))
        t.start()
        try:
            assert c.in_window.wait(20), "the restore never reached the window"
            c.clear(disk=True)                        # the user purges mid-restore
        finally:
            c.resume.set()
            t.join(20)
        alive = "k" in c._ram or c.get("k") is not None
        r.ok("A2: a frame cleared during a restore stays cleared") if not alive else \
            r.fail("A2 resurrection",
                   f"'k' back in _ram={'k' in c._ram}, get() serves pixels after clear()")


# ── A3 ────────────────────────────────────────────────────────────────────────

def test_v0332_a3_clear_does_not_orphan_a_frame_spilled_during_its_walk(r):
    """`clear(disk=True)`'s unlink loop runs unlocked (N `os.remove`s must not block every
    concurrent `get`), so a spill can land during it — legitimately, because a write that
    STARTED after the generation bump is post-clear content the generation check passes by
    design. The tail then rebound `_spilled` to a definite empty set and `_disk_bytes` to 0.

    A definite absence is the one wrong answer this file never gives: `_restore` short-circuits
    on the membership set without stat-ing, so the frame is on disk, unserveable, unreachable
    by the epoch cleanup, and its bytes leak the disk budget until something forces a rescan.
    Measured 66085 -> 0 with the file present. The witness is the spill counter, exactly as in
    `reindex_disk`: if it moved during the walk, stay UNKNOWN."""
    with tempfile.TemporaryDirectory() as d:
        c = tex_results.ResultCache(budget_mb=0, cache_dir=d)
        c.get("warmup_total_miss")            # membership becomes a LEARNED empty set
        assert c._spilled == set(), "setup: membership should be a learned empty set"
        rdir = os.path.realpath(c._spill_dir())

        gate_in, resume = threading.Event(), threading.Event()
        state, real_listdir = {"armed": True}, os.listdir

        def gated_listdir(p):
            out = real_listdir(p)
            if state["armed"] and os.path.realpath(str(p)) == rdir:
                state["armed"] = False        # clear's locked head is done; its snapshot is empty
                gate_in.set()
                assert resume.wait(20), "gate timeout"
            return out

        def clearer():
            os.listdir = gated_listdir
            try:
                c.clear(disk=True)
            finally:
                os.listdir = real_listdir

        t = threading.Thread(target=clearer)
        t.start()
        K = "k_postclear"
        try:
            assert gate_in.wait(20), "clear never reached its unlink walk"
            c.put(K, _flat(3.0))
            c.put("filler", _flat(0.0))       # evicts K -> a post-clear spill writes + indexes it
            assert os.path.exists(os.path.join(rdir, K + ".frame")), "setup: spill did not land"
        finally:
            resume.set()
            t.join(20)
        got = c.get(K)
        r.ok("A3: a frame spilled during clear's unlink walk is still serveable") \
            if got is not None and float(got.float().mean()) == 3.0 else \
            r.fail("A3 clear tail rebind",
                   f"get(K)={None if got is None else float(got.float().mean())}, "
                   f"_spilled={c._spilled!r}, _disk_bytes={c._disk_bytes!r}")


# ── A4 ────────────────────────────────────────────────────────────────────────

def test_v0332_a4_the_stored_quality_tag_survives_the_disk_tier(r):
    """The tier tag a frame was STORED under is now an entry slot and a `.frame` field. Two
    things need it that the packed dtype cannot supply: `patch_region` must not launder a
    preview base, and a requalify-on-idle pass has to ENUMERATE preview entries. `orig_dtype`
    is not a substitute — a preview frame that is unpackable (out of fp16 range, or a MASK)
    stores full and would read back as final."""
    with tempfile.TemporaryDirectory() as d:
        c = tex_results.ResultCache(budget_mb=0, cache_dir=d)
        c.put("p", _frame(res=64), quality=tex_packing.PREVIEW)
        assert c._ram["p"].quality == tex_packing.PREVIEW, "setup: put did not record the tag"
        c.put("q", _frame(res=8))                     # evicts "p" to disk
        assert c.get("p") is not None, "setup: the spilled frame should restore"
        tag = c._ram["p"].quality
        r.ok("A4: the tier tag round-trips through the spill tier") \
            if tag == tex_packing.PREVIEW else \
            r.fail("A4 tag lost on restore", f"restored entry carries quality={tag!r}")


def test_v0332_a4_patch_region_cannot_launder_a_preview_base(r):
    """PREVIEW IS VIRAL, and this is the one seam where the cache can SEE the upstream: the
    base is an entry with a recorded tag, not an opaque SHA. Patching an fp16-stored base while
    passing `quality=None` produced a result stored full-fp32 under a final-shaped key whose
    out-of-window bytes still carried fp16 quantization — mixed fidelity, silently (1.89e-03,
    half the 8-bit display quantum, which is exactly the size that survives review)."""
    with tempfile.TemporaryDirectory() as d:
        c = tex_results.ResultCache(cache_dir=d)
        c.put("base", _frame(res=32), quality=tex_packing.PREVIEW)
        assert c._ram["base"].tensor.dtype is torch.float16, "setup: base should be packed fp16"
        got = c.patch_region("mix", torch.full((1, 8, 8, 4), 0.25), (4, 4, 8, 8, 32, 32),
                             base_key="base", quality=None)
        e = c._ram.get("mix")
        ok = got is not None and e is not None and e.quality == tex_packing.PREVIEW
        r.ok("A4: a patch inherits its base's tier — no final-shaped mixed-fidelity frame") \
            if ok else \
            r.fail("A4 patch laundering",
                   f"got={got is not None} quality={None if e is None else e.quality!r} "
                   f"dtype={None if e is None else e.tensor.dtype}")


def test_v0332_a4_the_viral_rule_reaches_the_disk_tier_and_spares_bare_bases(r):
    """The first draft of A4 read `self._ram.get(base_key or key)` BEFORE the `get` that
    materializes the base, and both halves of that were wrong. Measured, both:

      * a preview base sitting on the DISK tier is not in `_ram` yet, so no tag propagated and
        it was laundered exactly as before the fix (`quality=None`, stored fp32). The `get` is
        what restores the entry — the question was being asked one step too early.
      * on the explicit-`base=` path with no `base_key=`, `base_key or key` resolves to the
        DESTINATION key. A stale preview entry under it forced every later patch of that key to
        preview, permanently, because `propagate_quality` only ratchets down. The documented
        host shape walks straight into this: `patch_region(f"s{i}", out, served, base=canvas[i])`
        re-patches one stable key per stage on every edit.

    Three cases, because the rule is only correct if it fires on exactly the right one."""
    with tempfile.TemporaryDirectory() as d:
        c = tex_results.ResultCache(cache_dir=d, budget_mb=0)   # every put evicts the previous
        c.put("base", _frame(res=32), quality=tex_packing.PREVIEW)
        c.put("filler", _frame(res=8))                          # forces "base" out to disk
        assert "base" not in c._ram and c.spills >= 1, "setup: base should have spilled"
        c.patch_region("spilled", torch.zeros(1, 8, 8, 4), (4, 4, 8, 8, 32, 32),
                       base_key="base")
        spilled = c._ram.get("spilled")

    with tempfile.TemporaryDirectory() as d:
        c = tex_results.ResultCache(cache_dir=d)
        c.put("s0", _frame(res=32), quality=tex_packing.PREVIEW)   # stale tag on the DESTINATION
        c.patch_region("s0", torch.zeros(1, 8, 8, 4), (4, 4, 8, 8, 32, 32),
                       base=_flat(0.5, res=32))                    # a bare, untagged base tensor
        bare = c._ram.get("s0")

    ok = (spilled is not None and spilled.quality == tex_packing.PREVIEW
          and bare is not None and bare.quality is None)
    r.ok("A4: the viral rule reaches a spilled base and does not invent one for a bare tensor") \
        if ok else \
        r.fail("A4 tag lookup",
               f"spilled base -> {None if spilled is None else spilled.quality!r} (want 'preview'); "
               f"bare base -> {None if bare is None else bare.quality!r} (want None)")


# ── A5 ────────────────────────────────────────────────────────────────────────

def test_v0332_a5_propagate_quality_accepts_a_single_tag(r):
    """`tuple("preview")` is `('p','r','e',...)`, which contains no PREVIEW — so the
    single-upstream spelling every caller reaches for first returned the UNSAFE answer while
    the list spelling returned the safe one. A rule whose safety depends on the caller's
    bracket choice is not a rule."""
    bare = tex_packing.propagate_quality(None, tex_packing.PREVIEW)
    listed = tex_packing.propagate_quality(None, [tex_packing.PREVIEW])
    r.ok("A5: one upstream tag propagates the same bare as bracketed") \
        if bare == listed == tex_packing.PREVIEW else \
        r.fail("A5 propagate_quality(str)", f"bare={bare!r} vs bracketed={listed!r}")


def test_v0332_a5_a_future_frame_format_is_refused(r):
    """`fmt` was written and never read: a record from a NEWER TEX was decoded best-effort and
    served as pixels. A forward-compatible reader cannot know what a future field means, so the
    only safe read of "newer than me" is to decline. (An ABSENT fmt still reads as v0 — that is
    the backward direction, which genuinely is decodable.)"""
    with tempfile.TemporaryDirectory() as d:
        c = tex_results.ResultCache(cache_dir=d)
        rec = {"t": torch.randint(-32768, 32767, (1, 16, 16, 4), dtype=torch.int16),
               "fmt": tex_results._FRAME_FORMAT + 1, "device": "cpu", "canvas": None,
               "epoch": tex_results.env_epoch(), "orig": "float32", "viewed": "uint16"}
        with open(c._disk_path("future"), "wb") as fh:
            pickle.dump(rec, fh)
        served = c.get("future")

    with tempfile.TemporaryDirectory() as d:
        c2 = tex_results.ResultCache(cache_dir=d)
        v0 = {"t": _frame(res=16), "device": "cpu", "canvas": None,
              "epoch": tex_results.env_epoch()}
        with open(c2._disk_path("v0"), "wb") as fh:
            pickle.dump(v0, fh)
        back = c2.get("v0")
    ok = served is None and back is not None and torch.equal(back, v0["t"])
    r.ok("A5: a future .frame format is refused; a v0 record still reads") if ok else \
        r.fail("A5 fmt is write-only",
               f"future record served={served is not None}, v0 compat={back is not None}")


def test_v0332_a5_patch_region_refuses_a_mismatch_instead_of_raising(r):
    """The window describes the SPATIAL extent only, so a base and a patch disagreeing on batch
    or channels reached the assignment and raised a RuntimeError out of a method whose whole
    contract is "refuse by returning None". Refusing is not a nicety: `patch_region` is the call
    a host makes when it is UNSURE a region cook is serviceable."""
    with tempfile.TemporaryDirectory() as d:
        c = tex_results.ResultCache(cache_dir=d)
        c.put("base", _flat(0.5, res=32))
        outcomes = {}
        for name, patch in (("channels", torch.zeros(1, 8, 8, 3)),
                            ("batch", torch.zeros(2, 8, 8, 4))):
            try:
                outcomes[name] = c.patch_region(f"m_{name}", patch, (4, 4, 8, 8, 32, 32),
                                                base_key="base")
            except Exception as exc:
                outcomes[name] = exc
    ok = all(v is None for v in outcomes.values())
    r.ok("A5: patch_region refuses a batch/channel mismatch with None") if ok else \
        r.fail("A5 patch_region raises", "; ".join(f"{k}: {v!r}" for k, v in outcomes.items()))


def test_v0332_a5_promote_keeps_a_patched_frame_on_its_home_device(r):
    """A5(d) was WITHDRAWN, and this row pins the reason so it is not re-attempted blind.

    The withdrawn change made `_promote` return the un-promoted host copy whenever the lock was
    held, to keep an 11.1 ms H2D out of a composite's critical section. But `_promote` is not a
    drain: a drain DEFERS (the queue survives, `patch_region` runs it on release) while that
    DEGRADED, with no `_pending_promotes` to make good on it. `frame[...] = patch` accepts a
    CUDA source into a CPU destination, so nothing raised — the patched frame was simply stored
    from a host buffer, `_admit` recorded `home="cpu"` for the fresh key, and the frame left the
    residency ladder for good, taking every stage downstream with it. Measured: result
    `device=cpu`, `home=cpu`, `promotions=0`.

    So the pin is on the INVARIANT the withdrawal restores — a patch whose base was demoted
    comes back on its home device — not on the mechanism. Whatever v0.34 does with a promote
    queue, this must stay true."""
    if "cuda" not in _devices():
        r.skip("A5 patched frame stays home-resident", "no CUDA on this box — nothing to demote")
        return
    with tempfile.TemporaryDirectory() as d:
        c = tex_results.ResultCache(cache_dir=d)
        c.put("base", _frame(res=64, device="cuda"))
        c.put("other", _frame(res=64, device="cuda", scale=0.5))
        c.set_vram_budget(0)
        base = c._ram.get("base")
        if base is None or tex_results._dev_bucket(base.device) != "cpu":
            r.fail("A5 patched frame stays home-resident",
                   f"setup: base was not demoted (device={None if base is None else base.device})")
            return
        out = c.patch_region("mix", torch.zeros(1, 8, 8, 4, device="cuda"),
                             (4, 4, 8, 8, 64, 64), base_key="base")
        e = c._ram.get("mix")
        ok = (out is not None and e is not None
              and tex_results._dev_bucket(e.home) == "cuda"
              and tex_results._dev_bucket(e.device) == "cuda")
        r.ok("A5: a patch over a demoted base stays on its home device") if ok else             r.fail("A5 patched frame stays home-resident",
                   f"out={out is not None} device={None if e is None else e.device} "
                   f"home={None if e is None else e.home} promotions={c.promotions}")


def test_v0332_a5_disarming_residency_cancels_queued_demotions(r):
    """`set_vram_budget(None)` documents itself as "v0.32's behaviour exactly", and v0.32 never
    moves a frame between devices — but victims queued under the old ceiling kept draining
    after it returned (measured `demotions` 0 -> 1 with `vram_budget_bytes=None`). Only the
    residency tier queues these, so dropping them cannot strand a governor request."""
    if "cuda" not in _devices():
        r.skip("A5 disarm cancels queued demotions", "no CUDA on this box — nothing to demote")
        return
    with tempfile.TemporaryDirectory() as d:
        c = tex_results.ResultCache(cache_dir=d)
        c.put("a", _frame(res=64, device="cuda"))
        c.put("b", _frame(res=64, device="cuda", scale=0.5))
        with c._lock:                                 # arm without enforcing or draining
            c._vram_budget = 0
            c._enforce_residency()
        assert c._pending_demotes, "setup: nothing queued at vram_budget=0"
        c.set_vram_budget(None)                       # disarm with the queue non-empty
        resident = all(tex_results._dev_bucket(e.device) == "cuda" for e in c._ram.values())
        ok = not c._pending_demotes and c.demotions == 0 and resident
    r.ok("A5: disarming residency cancels the demotions it had queued") if ok else \
        r.fail("A5 disarm semantics",
               f"pending={len(c._pending_demotes)} demotions={c.demotions} resident={resident}")


def test_v0332_a5_disarming_also_cancels_a_demotion_already_in_flight(r):
    """The row above only covers the QUEUE half. `set_vram_budget(None)` empties
    `_pending_demotes`, but a victim a drain has ALREADY popped is in neither the queue nor the
    clear's reach — it is mid-`egress`, holding only a local reference — so the commit-time
    `if self._vram_budget is None` re-check is what stops it, and nothing pinned that.

    It is the same argument `clear()` spells for spills ("a victim `_drain_spills` ALREADY
    popped is not in either queue, so clearing them does not stop it"), and without a row here
    the re-check reads as redundant belt-and-braces and gets deleted with the suite still green.

    Deterministic: the drain is blocked INSIDE its copy by a patched `egress`, the disarm runs
    while it is parked there, and only then is it released."""
    if "cuda" not in _devices():
        r.skip("A5 disarm cancels an in-flight demotion", "no CUDA on this box — nothing to demote")
        return
    from TEX_Wrangle.tex_runtime import streams
    with tempfile.TemporaryDirectory() as d:
        c = tex_results.ResultCache(cache_dir=d)
        c.put("a", _frame(res=128, device="cuda"))
        c.put("b", _frame(res=128, device="cuda", scale=0.5))

        gate_in, resume = threading.Event(), threading.Event()
        real_egress, state = streams.egress, {"armed": True}

        def gated_egress(src, **kw):
            if state["armed"]:
                state["armed"] = False
                gate_in.set()
                assert resume.wait(20), "gate timeout"
            return real_egress(src, **kw)

        streams.egress = gated_egress
        t = threading.Thread(target=lambda: c.set_vram_budget(0))   # arm -> queue -> drain
        t.start()
        try:
            assert gate_in.wait(20), "the demote never reached the gate"
            c.set_vram_budget(None)           # disarm while the D2H is in flight
        finally:
            resume.set()
            t.join(20)
            streams.egress = real_egress
        resident = all(tex_results._dev_bucket(e.device) == "cuda" for e in c._ram.values())
        ok = c.demotions == 0 and resident and not c._demoting
    r.ok("A5: a demotion already in flight is dropped at commit when the tier disarms") \
        if ok else \
        r.fail("A5 in-flight disarm",
               f"demotions={c.demotions} resident={resident} in_flight={c._demoting}")


# ── H: the pre-release bug hunt's findings ────────────────────────────────────
#
# Found by an adversarial hunt over the v0.33.2 tree AFTER Parts A/B/C were complete and before
# the tag. Two of them are this release's own headline fixes failing on an interleaving their
# pins could not reach — which is the whole argument for hunting a tree you believe is finished.

def test_v0332_h1_the_spill_ticket_orders_by_put_not_by_drain_start(r):
    """A1's ticket was claimed inside `_spill`, but `_drain_spills` pops the victim under
    `_lock` and RELEASES it before calling `_spill`. Two lock acquisitions, so pop order did not
    imply ticket order, and A1's own defect survived A1's fix.

    Interleaving: thread A pops (K, v1) and is descheduled in that gap; the main thread runs
    `put(K, v2)` and its entire spill, taking ticket 1; A resumes, takes ticket 2 holding the
    OLDER pixels, passes every check, and overwrites the winner. `get(K)` then serves v1
    forever — after both puts and both spills returned.

    The shipped A1 rows could not see it: their gate sits inside `egress`, which `_spill`
    reaches AFTER the old claim site, so the older writer already held the lower ticket and
    correctly bailed. This one pauses where a real scheduler can pause."""
    from TEX_Wrangle.tex_results import ResultCache

    class Gap(ResultCache):
        """Parks the first spill of K at `_spill`'s entry — exactly the pop/claim gap. The
        override only waits, then delegates to the shipped `_spill` unchanged."""

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.trip = False
            self.at_gap, self.resume = threading.Event(), threading.Event()

        def _spill(self, key, entry, seq):
            if self.trip and key == "k":
                self.trip = False
                self.at_gap.set()
                self.resume.wait(20)
            return super()._spill(key, entry, seq)

    with tempfile.TemporaryDirectory() as d:
        c = Gap(budget_mb=0, cache_dir=d)
        c.put("k", _flat(1.0))
        c.trip = True
        t = threading.Thread(target=lambda: c.put("f1", _flat(0.0)))
        t.start()
        try:
            assert c.at_gap.wait(20), "the v1 spill never reached the pop/claim gap"
            c.put("k", _flat(2.0))          # the blessed replace, with new pixels
            c.put("f2", _flat(0.0))         # evicts K(v2) -> spills it, claiming its ticket
        finally:
            c.resume.set()
            t.join(20)
        got = c.get("k")
        mean = None if got is None else float(got.float().mean())
    r.ok("H1: the spill ticket is ordered by the put that evicted, not by which drain starts") \
        if mean == 2.0 else \
        r.fail("H1 ticket inversion", f"disk served {mean} after put(K, 2.0) completed")


def test_v0332_h2_a_restore_cannot_outrun_clears_unlink_walk(r):
    """A2's residual. `clear(disk=True)` bumps the generation under the lock and then walks the
    directory UNLOCKED — N `os.remove`s must not block every concurrent `get`. A restore that
    captures `gen` AFTER the bump therefore MATCHES the current generation, reads a `.frame` the
    walk has not reached yet, and re-admits the frame the user just purged: alive in RAM, and
    back on disk at the next eviction.

    The generation alone cannot express this, because the restore is not from a stale
    generation — it is CONCURRENT with the purge. `_purging` says so for the walk's duration."""
    from TEX_Wrangle.tex_results import ResultCache

    class Slow(ResultCache):
        """Parks `clear`'s walk after its locked head has run. Only waits; no logic altered."""

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.hold = False
            self.in_walk, self.go = threading.Event(), threading.Event()

        def _spill_dir(self):
            out = super()._spill_dir()
            if self.hold:
                self.hold = False
                self.in_walk.set()
                self.go.wait(20)
            return out

    with tempfile.TemporaryDirectory() as d:
        c = Slow(budget_mb=0, cache_dir=d)
        c.put("k", _flat(7.0))
        c.put("f", _frame(res=8))                      # evicts "k" to the disk tier
        assert c.spills >= 1, "setup: k should have spilled"
        c.hold = True
        t = threading.Thread(target=lambda: c.clear(disk=True))
        t.start()
        try:
            assert c.in_walk.wait(20), "clear never reached its unlink walk"
            c.get("k")                                 # the restore lands mid-walk
        finally:
            c.go.set()
            t.join(20)
        alive = "k" in c._ram or c.get("k") is not None
    r.ok("H2: a restore concurrent with clear's walk cannot re-admit the purged frame") \
        if not alive else \
        r.fail("H2 clear purge window", "the cleared frame is alive after clear(disk=True)")


def test_v0332_h3_the_ratchet_never_packs_a_frame_put_refused_to_pack(r):
    """A4's ratchet took the base's TAG as licence to pack, and the tag is not that authority.
    `choose_storage` refuses several preview-tagged frames while the entry keeps the tag anyway:
    a MASK or LATENT (data planes are never packed), and any frame outside the fp16 range, both
    store full while reading `quality='preview'`. `patch_region` has no `kind` parameter, so its
    nested `put` passes `kind=None` and the ratcheted tag quantized a mask the host had
    explicitly protected — through a rule written to PREVENT fidelity loss.

    Gated on the stored REPRESENTATION instead, the rule is exactly as strong as its
    justification: a patch is never more faithful than its base, and never less faithful."""
    with tempfile.TemporaryDirectory() as d:
        c = tex_results.ResultCache(cache_dir=d)
        c.put("m", torch.rand(1, 32, 32, 1), quality=tex_packing.PREVIEW, kind="MASK")
        base = c._ram["m"]
        assert base.tensor.dtype is torch.float32 and base.orig_dtype is None, \
            "setup: a MASK must never be packed by put"
        c.patch_region("mp", torch.zeros(1, 8, 8, 1), (4, 4, 8, 8, 32, 32), base_key="m")
        mask_patch = c._ram.get("mp")
        # ...and a genuinely PACKED base must still ratchet, or the fix has gutted A4.
        c.put("b", _frame(res=32), quality=tex_packing.PREVIEW)
        c.patch_region("bp", torch.zeros(1, 8, 8, 4), (4, 4, 8, 8, 32, 32), base_key="b")
        packed_patch = c._ram.get("bp")
        ok = (mask_patch is not None and mask_patch.tensor.dtype is torch.float32
              and packed_patch is not None and packed_patch.tensor.dtype is torch.float16)
    r.ok("H3: the ratchet packs a reduced base's patch and never a data-kind one") if ok else \
        r.fail("H3 ratchet packs a refused frame",
               f"mask patch={None if mask_patch is None else mask_patch.tensor.dtype} "
               f"(want float32); packed patch="
               f"{None if packed_patch is None else packed_patch.tensor.dtype} (want float16)")


def test_v0332_h4_learn_spilled_does_not_orphan_a_racing_spill(r):
    """The THIRD unlocked directory walk in this class, and the one neither P0-6 nor A3 closed.
    `_spill` records into `_spilled` only when membership is already KNOWN, so a frame that
    spills during `_learn_spilled`'s scandir is recorded nowhere — and asserting a definite set
    afterwards forgets it. `_restore` then short-circuits on that set without stat-ing: the
    frame is on disk, unserveable forever, uncounted forever, and unreachable by the epoch
    cleanup. Same witness as the other two walks: if the spill counter moved, stay UNKNOWN."""
    with tempfile.TemporaryDirectory() as d:
        c = tex_results.ResultCache(budget_mb=0, cache_dir=d)
        c.put("seed", _frame(res=8))
        c.put("evict", _frame(res=8))
        real_scandir = os.scandir
        state = {"armed": True}

        def racing_scandir(path):
            it = real_scandir(path)
            if state["armed"]:
                state["armed"] = False
                c.put("racer", _flat(5.0, res=32))
                c.put("push", _frame(res=8))       # evicts racer -> it spills during the walk
            return it

        c._spilled = None
        os.scandir = racing_scandir
        try:
            c._learn_spilled()
        finally:
            os.scandir = real_scandir
        served = c.get("racer")
    r.ok("H4: a frame spilled during _learn_spilled's scan is still serveable") \
        if served is not None else \
        r.fail("H4 _learn_spilled race",
               f"the racing frame is unserveable; _spilled={c._spilled!r}")


def test_v0332_h5_a_failed_spill_write_is_not_counted_or_indexed(r):
    """`atomic_write` reports failure by RETURN — "so a caller keeping a running byte total can
    tell" — and `_atomic_pickle` discarded it. `_spill` IS that caller: on a full or read-only
    disk it advanced `spills`, added the key to `_spilled`, and adjusted `_disk_bytes` for a
    file that was never written. Worse than a miscount: if a PREVIOUS `.frame` for that key is
    still on disk, the key stays indexed and that stale record is served as current — A1's
    failure mode reached through a full disk rather than through a race."""
    with tempfile.TemporaryDirectory() as d:
        c = tex_results.ResultCache(budget_mb=0, cache_dir=d)
        real = tex_results._atomic_pickle
        tex_results._atomic_pickle = lambda _p, _d: False        # a disk that refuses writes
        try:
            c.put("k", _flat(3.0, res=32))
            c.put("f", _frame(res=8))                            # evicts k -> the write fails
        finally:
            tex_results._atomic_pickle = real
        ok = c.spills == 0 and "k" not in (c._spilled or set())
    r.ok("H5: a refused spill write advances no counter and indexes no key") if ok else \
        r.fail("H5 failed write counted",
               f"spills={c.spills} indexed={'k' in (c._spilled or set())}")


def test_v0332_h6_a_disarmed_cache_issues_no_demotion_copy(r):
    """Keeps the "residency runs even when disarmed" mutation KILLABLE.

    That row was killed until v0.33.2, and then A5's commit-time re-check
    (`if self._vram_budget is None: continue` in `_drain_demotes`) started masking it: with the
    `_enforce_residency` guard removed, victims are still QUEUED and their D2H is still issued,
    but the commit drops them — so `demotions`, the devices and the byte totals all look
    correct and no test failed. A guard nothing can observe is a guard the next reader deletes.

    Defense in depth is the right design here (one guard avoids the work, the other guarantees
    the outcome), so the answer is to observe the layer that would otherwise go quiet.

    Two things this row had to get right, both of which the earlier drafts got wrong:

      * the frames must be on CUDA. `_queue_demotions` only ever considers CUDA-resident
        entries, so a CPU-frame version leaves the mutation invisible for a second reason.
      * inspecting `_pending_demotes` after `put` returns cannot see it either — the drain runs
        on the way out and empties the queue, dropping each victim at the commit check.

    So what is counted is whether the D2H was ISSUED. That is also the thing actually worth
    protecting: the guard exists to avoid a full-frame copy per put, not to tidy a deque."""
    if "cuda" not in _devices():
        r.skip("H6 disarmed cache issues no D2H",
               "no CUDA on this box - only CUDA entries are ever queued for demotion")
        return
    from TEX_Wrangle.tex_runtime import streams
    with tempfile.TemporaryDirectory() as d:
        # A budget large enough that nothing spills, so every `egress` call is a demotion.
        c = tex_results.ResultCache(cache_dir=d, budget_mb=256)
        assert c._vram_budget is None, "setup: residency must be disarmed by default"
        real_egress, calls = streams.egress, {"n": 0}

        def counting_egress(src, **kw):
            calls["n"] += 1
            return real_egress(src, **kw)

        streams.egress = counting_egress
        try:
            for i in range(4):
                c.put(f"k{i}", _frame(res=64, device="cuda"))
            never_armed = calls["n"]
            # ...and after an explicit disarm, the other way to reach the same state.
            c.set_vram_budget(64)
            c.set_vram_budget(None)
            calls["n"] = 0
            for i in range(4):
                c.put(f"j{i}", _frame(res=64, device="cuda"))
            after_disarm = calls["n"]
        finally:
            streams.egress = real_egress
        ok = (never_armed == 0 and after_disarm == 0 and c.demotions == 0 and c.spills == 0)
    if ok:
        r.ok("H6: a disarmed cache issues no demotion copy - skipped, not merely dropped")
    else:
        r.fail("H6 disarmed cache does the work anyway",
               f"D2H issued: never-armed={never_armed} after-disarm={after_disarm}; "
               f"demotions={c.demotions} spills={c.spills}")


def test_v0332_h7_a_restore_that_starts_inside_a_purge_is_refused_at_capture(r):
    """H2's residual, found by the second cleanup pass and measured before fixing.

    H2's first draft checked the purge flag at `_admit` — the END of the restore window. The
    window a purge must cover starts at the FILE READ. So: `clear`'s locked head runs (generation
    0->1, purge depth 1); a restore starts, captures a generation that now MATCHES, and reads the
    `.frame`; the walk unlinks it and the tail drops the depth to 0; the restore reaches `_admit`,
    finds the generation matching and the flag already cleared, and re-admits the frame the user
    purged. Measured with the whole `clear(disk=True)` returned.

    The check now sits where the generation is captured, so a restore that starts inside the
    window is refused before it reads anything — which also deletes the doomed unpickle and H2D,
    and makes the `_admit` clause unreachable (it is gone; an unexercisable guard is one the next
    reader deletes for the wrong reason)."""
    from TEX_Wrangle.tex_results import ResultCache

    class Both(ResultCache):
        """Parks clear's walk AND the re-admit independently. Only waits; no logic altered."""

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.hold_walk = self.hold_admit = False
            self.in_walk, self.walk_go = threading.Event(), threading.Event()
            self.at_admit, self.admit_go = threading.Event(), threading.Event()

        def _spill_dir(self):
            out = super()._spill_dir()
            if self.hold_walk:
                self.hold_walk = False
                self.in_walk.set()
                self.walk_go.wait(20)
            return out

        def _admit(self, *a, **kw):
            if self.hold_admit:
                self.hold_admit = False
                self.at_admit.set()
                self.admit_go.wait(20)
            return super()._admit(*a, **kw)

    with tempfile.TemporaryDirectory() as d:
        c = Both(budget_mb=0, cache_dir=d)
        c.put("k", _flat(9.0, res=32))
        c.put("f", _frame(res=8))                      # evicts "k" to the disk tier
        assert c.spills >= 1, "setup: k should have spilled"
        c.hold_walk = True
        tc = threading.Thread(target=lambda: c.clear(disk=True))
        tc.start()
        tg = None
        try:
            assert c.in_walk.wait(20), "clear never parked in its walk"
            # The restore starts INSIDE the window: its generation will match.
            c.hold_admit = True
            tg = threading.Thread(target=lambda: c.get("k"))
            tg.start()
            # It may be refused at capture (the fix) and never reach `_admit` at all, so this
            # wait is allowed to time out — that outcome IS the fix working.
            reached_admit = c.at_admit.wait(3)
        finally:
            c.walk_go.set()
            tc.join(20)
            c.admit_go.set()
            if tg is not None:
                tg.join(20)
        alive = "k" in c._ram or c.get("k") is not None
    if not alive:
        r.ok("H7: a restore starting inside a purge is refused at capture, not after the walk")
    else:
        r.fail("H7 purge checked too late",
               f"the purged frame is alive after clear() returned "
               f"(restore reached _admit: {reached_admit})")


def test_v0332_h7_the_purge_depth_survives_an_interrupted_walk(r):
    """The purge marker is a DEPTH COUNT dropped in a `finally`, not a bool set and cleared.

    Two reasons, both real. `clear`'s walk is wrapped in `except Exception`, so a
    `BaseException` — a Ctrl-C during a large unlink walk is the realistic one — skipped the tail
    that cleared the flag, and every subsequent restore declined for the life of the process:
    the disk tier silently becomes a recook path with no counter or log saying so. And two
    threads in `clear(disk=True)` on one cache had the first to finish clear the marker out from
    under the other's walk, reopening H2 for the rest of it.

    A count also reads identically at the check site, so nothing else had to change."""
    from TEX_Wrangle.tex_results import ResultCache

    class Interrupted(ResultCache):
        def _spill_dir(self):
            out = super()._spill_dir()
            if getattr(self, "boom", False):
                self.boom = False
                raise KeyboardInterrupt("user hit Ctrl-C mid-walk")
            return out

    with tempfile.TemporaryDirectory() as d:
        c = Interrupted(budget_mb=0, cache_dir=d)
        c.put("k", _flat(4.0, res=32))
        c.put("f", _frame(res=8))
        assert c.spills >= 1, "setup: k should have spilled"
        c.boom = True
        try:
            c.clear(disk=True)
        except KeyboardInterrupt:
            pass
        depth = c._purge_depth
        # The tier must still work: put, evict, and read the frame back through the disk tier.
        c.put("later", _flat(6.0, res=32))
        c.put("push", _frame(res=8))
        served = c.get("later")
        ok = depth == 0 and served is not None and float(served.float().mean()) == 6.0
    r.ok("H7: an interrupted purge leaves the depth at 0 and the disk tier working") if ok else \
        r.fail("H7 purge depth stuck",
               f"_purge_depth={depth} after an interrupted walk; "
               f"disk tier serves={served is not None}")
