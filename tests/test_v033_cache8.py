"""CACHE-8 (v0.33) — deep cache tiers: residency, packing, and the codec that isn't there.

The item as written asks for two things and this file defends both, plus the negative result:

  RESIDENCY   A cold CUDA frame moves to host RAM instead of to disk, and comes back on reuse.
              The rows that matter are the ones about what must NOT change: the frame's pixels,
              its home device, and the per-device byte accounting the governor reads. A
              residency bug is silent by construction — the frame is still there, still
              servable, just on the wrong device with the wrong bytes charged — so every row
              here checks accounting alongside pixels.

  PACKING     uint16 is offered, never chosen automatically, and refuses out-of-range data
              rather than clipping it.

  THE CODEC   Measured and rejected. `benchmarks/cache_capacity_bench.py` is the record; the
              row here pins that no compression path was left switched on by accident.

  HINTS       A host that PREDICTS demand can steer the victim walks without reading:
              `touch(key)` ranks a resident entry just below the most recent one, and
              `key in cache` asks whether it is resident. Neither is a read, so the rows pin
              what they must NOT do — count, restore, promote, unpack, move a frame, reorder
              anything but the one entry, or change what an arbitration frees — beside what
              they do, and race both against every other door into the table.

              `touch_promote(key)` (CACHE-11) is `touch` plus ONE addition, ruled narrower than
              the promotion-on-a-hint `touch` still declines: if the entry is demoted, it comes
              home on the hint, counted in `promotions` and never in `hits`. Its rows are the
              same picture with one column changed.

Every row is CPU-safe: `_devices()` adds the CUDA rows when there is a GPU, and the residency
ladder degenerates honestly to one rung without one (there is nothing to demote FROM).
"""
import contextlib
import tempfile
import threading
import time
from collections import OrderedDict

import torch

from helpers import devices as _devices, make_gradient_frame as _frame
from TEX_Wrangle import tex_engine, tex_memory, tex_packing, tex_results


# ── residency ─────────────────────────────────────────────────────────────────

def test_v033_cache8_residency_is_off_until_armed(r):
    """v0.32's behaviour is the shipped behaviour. A cache that was never given a VRAM ceiling
    must not start moving frames between devices because the package was upgraded — the entire
    'off means off' half of the S-5 discipline, and the reason `balanced` carries None."""
    with tempfile.TemporaryDirectory() as d:
        c = tex_results.ResultCache(cache_dir=d)
        st = c.stats()
        f = _frame()
        for i in range(6):
            c.put(f"k{i}", f)
        st2 = c.stats()
        ok = (st["vram_budget_bytes"] is None and st2["demotions"] == 0
              and st2["promotions"] == 0 and st2["demoted"] == 0)
        r.ok("CACHE-8: an unarmed cache never demotes (v0.32 behaviour, unchanged)") if ok else \
            r.fail("CACHE-8 off", f"{st2}")


def test_v033_cache8_demote_frees_vram_and_keeps_the_frame(r):
    """The headline. Over the VRAM ceiling, the coldest frame moves to host RAM: the cuda byte
    bucket drops by exactly its size, the cpu bucket gains exactly the same, the entry stays
    servable, and the pixels are unchanged. Measured against the alternative it replaces —
    a disk spill at 77.9-78.8 ms versus 5.7-5.9 ms to demote (2048^2, two runs)."""
    if "cuda" not in _devices():
        r.skip("CACHE-8 demote", "no CUDA on this box - nothing to demote from")
        return
    with tempfile.TemporaryDirectory() as d:
        # 256² x4 fp32 = exactly 1 MB, so a budget expressed in whole MB can actually name a
        # frame count. At 64² the four frames together are a quarter of the smallest
        # representable budget, and nothing would ever be over it.
        c = tex_results.ResultCache(cache_dir=d)
        c.set_vram_budget(2)                                 # room for 2 of the 4 frames
        for i in range(4):
            c.put(f"k{i}", _frame(res=256, device="cuda", scale=1.0 + i * 0.01))
        # One snapshot, read once. An earlier draft compared `stats()` taken BEFORE a `get`
        # against `_bytes_by_dev` read after it — and the `get` promotes, so the two halves of
        # the identity described different moments and the row failed on a cache that was
        # working perfectly. Take the picture, then disturb it.
        st = c.stats()
        served = c.get("k0")
        entry = c._ram.get("k0")
        ok = (st["demotions"] >= 1 and st["demoted"] >= 1
              and st["vram_bytes"] <= st["vram_budget_bytes"]
              and st["ram_bytes"] > st["vram_bytes"]     # bytes really are on the host now
              and served is not None and entry is not None
              and entry.home.startswith("cuda"))
        r.ok(f"CACHE-8: {st['demotions']} frame(s) demoted; VRAM "
             f"{st['vram_bytes']} <= budget {st['vram_budget_bytes']}, all still servable") \
            if ok else r.fail("CACHE-8 demote", f"{st} entry={None if entry is None else (entry.device, entry.canvas, entry.orig_dtype, entry.home)}")


def test_v033_cache8_demoted_frame_is_bit_exact(r):
    """A demotion is a device move, not a representation change. If this row ever fails, the
    residency tier has become a lossy tier without anyone deciding that it should."""
    if "cuda" not in _devices():
        r.skip("CACHE-8 demote bit-exactness", "no CUDA on this box")
        return
    with tempfile.TemporaryDirectory() as d:
        src = _frame(res=64, device="cuda")
        c = tex_results.ResultCache(cache_dir=d)
        c.set_vram_budget(0)                                  # demote everything demotable
        c.put("a", src)
        c.put("b", _frame(res=64, device="cuda", scale=0.5))
        got = c.get("a")
        ok = got is not None and torch.equal(got.to("cuda").float(), src.float())
        r.ok("CACHE-8: a demoted frame round-trips bit-exact") if ok else \
            r.fail("CACHE-8 demote-exact",
                   f"maxdiff {float((got.to('cuda') - src).abs().max()) if got is not None else 'None'}")


def test_v033_cache8_promote_on_reuse_returns_it_home(r):
    """The other rung. A hit on a demoted frame promotes it back to the device it was cooked
    on, and the accounting follows. Serving a CUDA frame from the CPU forever would be a
    correctness-preserving performance bug — the worst kind to find later."""
    if "cuda" not in _devices():
        r.skip("CACHE-8 promote", "no CUDA on this box")
        return
    with tempfile.TemporaryDirectory() as d:
        c = tex_results.ResultCache(cache_dir=d)
        c.set_vram_budget(0)
        c.put("a", _frame(res=64, device="cuda"))
        c.put("b", _frame(res=64, device="cuda", scale=0.5))
        before = c.stats()
        got = c.get("a")
        entry = c._ram.get("a")
        after = c.stats()
        ok = (before["demoted"] >= 1 and after["promotions"] >= 1
              and got is not None and got.device.type == "cuda"
              and entry is not None and entry.device == entry.home
              and after["vram_bytes"] + c._bytes_by_dev["cpu"] == after["ram_bytes"])
        r.ok("CACHE-8: a hit on a demoted frame promotes it home and re-charges the bytes") \
            if ok else r.fail("CACHE-8 promote", f"before={before} after={after}")


def test_v033_cache8_a_spilled_demoted_frame_comes_back_to_its_home(r):
    """The interaction the two ladders have with each other, and the one place a home device
    can be lost: a frame demoted to RAM and THEN spilled to disk must restore as a CUDA frame.
    Writing slot 3 (where it is) instead of slot 6 (where it belongs) would turn the residency
    tier into a one-way trip to the CPU, discovered only under memory pressure."""
    if "cuda" not in _devices():
        r.skip("CACHE-8 demote+spill", "no CUDA on this box")
        return
    with tempfile.TemporaryDirectory() as d:
        f = _frame(res=64, device="cuda")
        # THE ORDER MATTERS, and an earlier draft got it wrong in a way that made this row
        # decorative: with `budget_mb=0` from the start, `_enforce_ram_budget` evicts the entry
        # before `_drain_demotes` ever runs, so the frame spills while still on CUDA — where
        # `device` and `home` are equal and the bug under test cannot exist. The mutation
        # "persist `device` instead of `home`" SURVIVED against that version.
        # So: demote first, with room to spare, and only then tighten the budget.
        c = tex_results.ResultCache(cache_dir=d, budget_mb=64)
        c.set_vram_budget(0)                                   # everything demotes to host RAM
        c.put("a", f)
        c.put("b", _frame(res=64, device="cuda", scale=0.5))
        demoted = c._ram["a"]
        really_demoted = demoted.device == "cpu" and demoted.home.startswith("cuda")
        c.set_budget(0)                                        # NOW force it out to disk
        # Make room again and disarm residency BEFORE reading it back. With the ceiling still
        # at 0 the restored frame is re-demoted the instant it lands — correct behaviour, and it
        # would mask the thing under test behind the budget doing its job.
        c.set_vram_budget(None)
        c.set_budget(64)
        got = c.get("a")
        entry = c._ram.get("a")
        ok = (really_demoted and c.spills >= 1
              and got is not None and entry is not None and entry.home.startswith("cuda")
              and got.device.type == "cuda"
              and float((got.to("cuda").float() - f.float()).abs().max()) == 0.0)
        r.ok("CACHE-8: a demoted-then-spilled frame restores to its HOME device, bit-exact") \
            if ok else r.fail("CACHE-8 demote+spill",
                              f"really_demoted={really_demoted} spills={c.spills} "
                              f"got={None if got is None else got.device} "
                              f"entry={None if entry is None else (entry.device, entry.home)}")


def test_v033_cache8_governor_prefers_demotion_over_eviction(r):
    """The CACHE-5 integration. When the governor asks for VRAM and the residency tier is
    armed, the bytes come back by MOVING frames, not by dropping them — so the eviction that
    used to cost the cache its contents now costs it a device hop, and the hit rate survives."""
    if "cuda" not in _devices():
        r.skip("CACHE-8 governor preference", "no CUDA on this box")
        return
    with tempfile.TemporaryDirectory() as d:
        f = _frame(res=64, device="cuda")
        nb = f.numel() * 4
        c = tex_results.ResultCache(cache_dir=d)
        c.set_vram_budget(10_000)                              # huge: no ambient demotion
        for i in range(4):
            c.put(f"k{i}", _frame(res=64, device="cuda", scale=1.0 + i * 0.01))
        freed = c.evict_bytes(2 * nb, dev_type="cuda")
        st = c.stats()
        served = sum(1 for i in range(4) if c.get(f"k{i}") is not None)
        ok = (freed >= 2 * nb and st["demotions"] >= 2 and c.evictions == 0
              and c.spills == 0 and served == 4)
        r.ok(f"CACHE-8: the governor's {freed} VRAM bytes came from {st['demotions']} "
             f"demotions, 0 evictions — all 4 frames still served") if ok else \
            r.fail("CACHE-8 governor", f"freed={freed} demotions={st['demotions']} "
                                       f"evictions={c.evictions} spills={c.spills} served={served}")


def test_v033_cache8_unarmed_governor_evicts_exactly_as_before(r):
    """The other side of that integration, and the row that keeps the previous one honest:
    with the tier disarmed the governor hook must behave exactly as v0.32 — spill, don't move."""
    if "cuda" not in _devices():
        r.skip("CACHE-8 unarmed governor", "no CUDA on this box")
        return
    with tempfile.TemporaryDirectory() as d:
        f = _frame(res=64, device="cuda")
        nb = f.numel() * 4
        c = tex_results.ResultCache(cache_dir=d)
        for i in range(4):
            c.put(f"k{i}", _frame(res=64, device="cuda", scale=1.0 + i * 0.01))
        freed = c.evict_bytes(2 * nb, dev_type="cuda")
        st = c.stats()
        ok = freed >= 2 * nb and st["demotions"] == 0 and c.evictions >= 2
        r.ok("CACHE-8: an unarmed cache still evicts-and-spills, byte for byte as v0.32") \
            if ok else r.fail("CACHE-8 unarmed governor",
                              f"freed={freed} demotions={st['demotions']} ev={c.evictions}")


# ── packing ───────────────────────────────────────────────────────────────────

def test_v033_cache8_uint16_is_offered_never_chosen(r):
    """uint16 is measurably better than fp16 inside [0,1] — 7.7e-6 against 2.4e-4, at the same
    two bytes — and it CLIPS outside. That is why it is an explicit request and never an
    automatic one: choosing it by sniffing a frame's range would be exactly the silent
    auto-tuning S-5 forbids, and would break the first HDR frame that arrived in range."""
    f = _frame(res=32)
    auto = tex_packing.choose_storage(f, quality=tex_packing.PREVIEW)
    asked = tex_packing.choose_storage(f, quality=tex_packing.PREVIEW, storage="uint16")
    hdr = tex_packing.choose_storage(_frame(res=32, scale=4.0),
                                     quality=tex_packing.PREVIEW, storage="uint16")
    final = tex_packing.choose_storage(f, quality=tex_packing.FINAL, storage="uint16")
    ok = (auto == tex_packing.FP16 and asked == tex_packing.UINT16
          and hdr is None and final is None)
    r.ok("CACHE-8: uint16 only on request, refuses out-of-range, never on the final tier") \
        if ok else r.fail("CACHE-8 uint16", f"auto={auto} asked={asked} hdr={hdr} final={final}")


def test_v033_cache8_uint16_beats_fp16_in_range(r):
    """The measurement that justifies offering a second codec at all, as a test rather than a
    docs claim. If uint16 ever stops being ~30x more accurate in [0,1], it has no reason to
    exist and should be deleted rather than documented."""
    with tempfile.TemporaryDirectory() as d:
        f = _frame(res=128)
        c = tex_results.ResultCache(cache_dir=d)
        c.put("h", f, quality=tex_packing.PREVIEW)
        c.put("u", f, quality=tex_packing.PREVIEW, storage="uint16")
        e16 = float((c.get("h") - f).abs().max())
        eu = float((c.get("u") - f).abs().max())
        same_size = c._ram["h"].nbytes == c._ram["u"].nbytes
        ok = eu * 10 < e16 and same_size
        r.ok(f"CACHE-8: uint16 err {eu:.2e} vs fp16 {e16:.2e} ({e16/max(eu,1e-12):.0f}x) at "
             f"identical size") if ok else \
            r.fail("CACHE-8 uint16 accuracy", f"u={eu:.3e} h={e16:.3e} same_size={same_size}")


def test_v033_cache8_no_compression_path_is_switched_on(r):
    """The negative result, pinned. The measured Pareto rejected every general-purpose codec:
    at 4K, zlib-1 costs 6685 ms to encode and 920 ms to DECODE against 332 ms to simply write
    the frame and 59 ms to read it back. Decode is paid on every hit, so this is not close.
    This row exists so a future 'small' addition of a codec to the spill path has to argue with
    the measurement instead of slipping past it."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent.parent / "tex_results.py").read_text(
        encoding="utf-8")
    hits = [n for n in ("zlib", "lzma", "bz2", "gzip", "compress") if n in src]
    r.ok("CACHE-8: no entropy codec on the cache path (the Pareto said don't)") if not hits \
        else r.fail("CACHE-8 codec", f"tex_results.py references {hits} — re-run "
                                     f"benchmarks/cache_capacity_bench.py before keeping it")


# ── GOV-1's new knob ──────────────────────────────────────────────────────────

def test_v033_cache8_profiles_carry_the_residency_ceiling(r):
    """The knob the v0.32 item text reserved for 'compression aggressiveness'. It is a
    residency ceiling instead, because that is what the measurement said buys capacity — and
    `balanced` carries None so the shipped default is still residency OFF."""
    want = {"performance": 2048, "balanced": None, "efficient": 256}
    got = {n: tex_memory.profile_knobs(n).get("vram_mb") for n in tex_memory.profiles()}
    ordered = got["efficient"] < got["performance"]
    r.ok(f"CACHE-8: profiles carry vram_mb {got}") if got == want and ordered else \
        r.fail("CACHE-8 profile knob", f"{got} != {want}")


def test_v033_cache8_profile_reaches_and_restores_the_ceiling(r):
    """The bug GOV-1 already had once, one knob over: a preset that can SET a value but not
    put it back leaves the cache enforcing `efficient` while `tex doctor` reports `balanced`.
    `_armed_caches` had to become a dict of remembered defaults for this to be possible."""
    tex_memory._reset_profile_for_test()
    try:
        with tempfile.TemporaryDirectory() as d:
            c = tex_results.ResultCache(cache_dir=d)
            shipped = c._vram_budget
            tex_memory.register_result_cache(c, name="t_v033_gov")
            tex_memory.set_profile("efficient")
            tight = c._vram_budget
            tex_memory.set_profile("performance")
            loose = c._vram_budget
            tex_memory.set_profile("balanced")
            back = c._vram_budget
            ok = (shipped is None and tight == 256 << 20 and loose == 2048 << 20
                  and back is None)
            r.ok("CACHE-8: the ceiling reaches an armed cache and `balanced` restores OFF") \
                if ok else r.fail("CACHE-8 gov restore",
                                  f"shipped={shipped} tight={tight} loose={loose} back={back}")
            tex_memory.get_cache_registry().unregister("t_v033_gov")
    finally:
        tex_memory._reset_profile_for_test()


def test_v033_cache8_is_absent_from_the_default_comfyui_path(r):
    """Invariant #7 as a source canary. Residency moves frames between devices; the default
    ComfyUI cook must be unable to reach the code that does it, which a grep can decide and a
    timing cannot. `touch` steers that code's victim choice, so it is on the list too, and so is
    `touch_promote` (CACHE-11), the one hint that can actually promote."""
    import pathlib
    node = (pathlib.Path(__file__).resolve().parent.parent / "tex_node.py").read_text(
        encoding="utf-8")
    hits = [n for n in ("set_vram_budget", "residency", "_demote", "_promote", "touch",
                        "touch_promote") if n in node]
    r.ok("CACHE-8: tex_node.py cannot reach the residency ladder") if not hits else \
        r.fail("CACHE-8 invariant#7", f"tex_node.py mentions {hits}")


# ── residency hints: `touch` and `in` are not reads ───────────────────────────
#
# `get` is a READ: it counts a hit or a miss, promotes a demoted frame, falls through to a disk
# restore, and unpacks a preview-stored one. A host that only PREDICTS demand wants the victim
# walks to reach a frame later and nothing else, so a hint has its own two doors — and every row
# below pins what they must not do beside what they do.

#: Every method a read, a residency move or a spill goes through. A hint that reaches one of
#: them is doing a read's work.
_READ_DOORS = ("get", "put", "_admit", "_remove", "_restore", "_learn_spilled", "_promote",
               "_queue_demotions", "_enforce_residency", "_enforce_ram_budget",
               "_drain_demotes", "_drain_spills", "_spill", "_claim_spill_ticket")


@contextlib.contextmanager
def _doors_spied(c):
    """Count calls into `_READ_DOORS` on `c`, and into the three module-level functions a read
    reaches outside the class (the version check, the unpack, the D2H), without changing what
    any of them does. Instance attributes shadow the methods — so `self._restore` inside the
    cache resolves to the counter — and everything is put back on exit."""
    from TEX_Wrangle.tex_runtime import streams
    calls: dict = {}

    def counted(label, fn):
        def spy(*a, **kw):
            calls[label] = calls.get(label, 0) + 1
            return fn(*a, **kw)
        return spy

    for name in _READ_DOORS:
        setattr(c, name, counted(name, getattr(c, name)))
    saved = [(m, n, getattr(m, n)) for m, n in
             ((tex_engine, "verify_unmutated"), (tex_packing, "unpack"), (streams, "egress"))]
    for m, n, fn in saved:
        setattr(m, n, counted(n, fn))
    try:
        yield calls
    finally:
        for name in _READ_DOORS:
            c.__dict__.pop(name, None)
        for m, n, fn in saved:
            setattr(m, n, fn)


def _hint_state(c):
    """What a hint must leave EXACTLY as it found it: every entry's slots (its tensor by
    identity, so a copy or a device move shows), the byte buckets, both drain queues, the
    in-flight demotions, the disk index and write tickets, and every `stats()` counter. Keyed by
    entry, so the LRU ORDER is deliberately not in it, and `touches` is left out too — those are
    the two things a touch exists to change, and the rows pin them separately."""
    with c._lock:
        return {
            "entries": {k: (id(e), id(e.tensor), e.stamp, e.nbytes, e.device, id(e.canvas),
                            e.orig_dtype, e.home, e.quality) for k, e in c._ram.items()},
            "bytes_by_dev": dict(c._bytes_by_dev),
            "pending_demotes": [(k, id(e)) for k, e in c._pending_demotes],
            "pending_spills": [(k, id(e), seq) for k, e, seq in c._pending_spills],
            "demoting": set(c._demoting),
            "spilled": None if c._spilled is None else set(c._spilled),
            "disk_bytes": c._disk_bytes,
            "spill_seq": dict(c._spill_seq),
            "generation": c._generation,
            "stats": {k: v for k, v in c.stats().items() if k != "touches"},
        }


def _changed(before, after):
    """The `_hint_state` fields that differ — empty on a pass, the failure message otherwise."""
    return sorted(k for k in before if before[k] != after[k])


def _spill_oldest(c):
    """Push the OLDEST entry out to the disk tier through the governor hook, and say whether it
    got there: spilled and not resident, the state in which a `get` would restore it."""
    key = next(iter(c._ram))
    spills = c.spills
    c.evict_bytes(c._ram[key].nbytes, dev_type=tex_results._dev_bucket(c._ram[key].device))
    return key not in c._ram and c.spills == spills + 1


def _recount(c):
    """(the per-device byte buckets recomputed from the entries, the buckets as maintained)."""
    with c._lock:
        recount = {"cuda": 0, "cpu": 0}
        for e in c._ram.values():
            recount[tex_results._dev_bucket(e.device)] += e.nbytes
        return recount, dict(c._bytes_by_dev)


def test_v033_cache8_touch_is_not_a_read(r):
    """A touch changes one entry's place in the order and the `touches` counter, and NOTHING
    else — pinned as a before/after picture that must be equal apart from those two, over every
    shape a key can have: resident at full precision; resident PREVIEW-stored (a `get` would
    unpack it); spilled to disk only (a `get` would restore it); and held by nobody (a `get`
    would count a miss). No read, residency or spill door is reached, and the spilled frame is
    still on disk and still served afterwards. In an EMPTY cache a touch is a plain False: there
    is no most recent entry to read, and that must not raise."""
    with tempfile.TemporaryDirectory() as d0:
        empty_ok = tex_results.ResultCache(cache_dir=d0).touch("anything") is False
    with tempfile.TemporaryDirectory() as d:
        c = tex_results.ResultCache(cache_dir=d, budget_mb=64)
        gone = _frame(res=32, scale=0.25)
        c.put("gone", gone)
        for i, k in enumerate("abc"):
            c.put(k, _frame(res=32, scale=1.0 + i * 0.01))
        c.put("prev", _frame(res=32, scale=0.5), quality=tex_packing.PREVIEW)
        c.put("d", _frame(res=32, scale=0.75))
        spilled = _spill_oldest(c)
        packed = c._ram["prev"].orig_dtype is not None
        before, t0 = _hint_state(c), c.stats().get("touches")
        with _doors_spied(c) as calls:
            got = {k: c.touch(k) for k in ("a", "prev", "gone", "nobody")}
        after, t1 = _hint_state(c), c.stats().get("touches")
        order = list(c._ram)
        back = c.get("gone")                  # outside the spy: the disk tier is still intact
        served = back is not None and torch.equal(back, gone) and c.restores == 1
    changed = _changed(before, after)
    ok = (empty_ok and spilled and packed and served and not calls and not changed
          and got == {"a": True, "prev": True, "gone": False, "nobody": False}
          and (t0, t1) == (0, 2) and order == ["b", "c", "a", "prev", "d"])
    r.ok("touch: reorders one entry and counts it in `touches` — no hit, miss, restore, "
         "promotion, unpack or queued work, for resident, preview, spilled and absent keys") \
        if ok else r.fail("touch non-effects",
                          f"returned={got} doors={calls} changed={changed} "
                          f"touches {t0}->{t1} order={order} empty_ok={empty_ok} "
                          f"spilled={spilled} packed={packed} served={served}")


def test_v033_cache8_touch_steers_the_victim_walks(r):
    """What a hint is FOR: a walk that takes victims oldest-first reaches a touched entry after
    every untouched one. Checked for the two walks every device has — the governor's
    `evict_bytes` and the RAM budget `put` enforces — each against an identical cache that was
    not touched, so the row cannot pass on a walk that would have spared the entry anyway. In
    both, the frame put last stays last: a touch ranks BELOW the most recent demand. (The
    residency walk is the CUDA leg of the demand-outranks-hint row.)"""
    def filled(d, keys, res, budget_mb):
        c = tex_results.ResultCache(cache_dir=d, budget_mb=budget_mb)
        for i, k in enumerate(keys):
            c.put(k, _frame(res=res, scale=1.0 + i * 0.01))
        return c

    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        hinted, control = filled(d1, "abcd", 32, 64), filled(d2, "abcd", 32, 64)
        nb = hinted._ram["a"].nbytes
        hinted.touch("a")
        ranked = list(hinted._ram)
        freed = (hinted.evict_bytes(nb, dev_type="cpu"), control.evict_bytes(nb, dev_type="cpu"))
        gov = (ranked, list(hinted._ram), list(control._ram), freed,
               (hinted.spills, control.spills))
    want_gov = (["b", "c", "a", "d"], ["c", "a", "d"], ["b", "c", "d"], (nb, nb), (1, 1))

    # 256² x4 fp32 is exactly 1 MiB, so a whole-MB budget names a frame count: three fit.
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        hinted, control = filled(d1, "abc", 256, 3), filled(d2, "abc", 256, 3)
        hinted.touch("a")
        for c in (hinted, control):
            c.put("d", _frame(res=256, scale=1.5))
        bud = (list(hinted._ram), list(control._ram), (hinted.evictions, control.evictions))
    want_bud = (["a", "c", "d"], ["b", "c", "d"], (1, 1))

    r.ok("touch: evict_bytes and the RAM budget both take an untouched entry first, and the "
         "frame put last stays last") if gov == want_gov and bud == want_bud else \
        r.fail("touch steering", f"evict_bytes {gov} != {want_gov}; budget {bud} != {want_bud}")


def test_v033_cache8_touch_leaves_the_demanded_frame_on_top(r):
    """Demand outranks a hint. With `x` put last, touching an older `w` leaves `x` the most
    recent entry, on every device. On CUDA the residency walk is then checked both ways the
    other walks are: a ceiling one frame too low demotes the coldest UNTOUCHED frame rather than
    `w`, and a ceiling of 0 demotes every other frame and still spares `x` by key (A7) — so a
    hint never hands the frame about to be read to the demotion walk."""
    for dev in _devices():
        with tempfile.TemporaryDirectory() as d:
            c = tex_results.ResultCache(cache_dir=d, budget_mb=64)
            res = 256 if dev == "cuda" else 32          # 1 MiB frames, so ceilings name frames
            for i, k in enumerate("wyzx"):
                c.put(k, _frame(res=res, device=dev, scale=1.0 + i * 0.01))
            c.touch("w")
            order = list(c._ram)
            ok, detail = order == ["y", "z", "w", "x"], f"order={order}"
            if dev == "cuda":
                c.set_vram_budget(3)                    # one frame over: ONE victim
                one = {k: c._ram[k].device for k in c._ram}
                c.set_vram_budget(0)                    # every frame over but the most recent
                zero = {k: c._ram[k].device for k in c._ram}
                ok = (ok and one["y"] == "cpu" and all(one[k].startswith("cuda") for k in "zwx")
                      and all(zero[k] == "cpu" for k in "yzw") and zero["x"].startswith("cuda")
                      and list(c._ram)[-1] == "x" and c.demotions == 3)
                detail += f" ceiling=3 {one} ceiling=0 {zero} demotions={c.demotions}"
        r.ok(f"[{dev}] touch: the frame put last keeps the top slot"
             + (", and the residency walk demotes around the hint" if dev == "cuda" else "")) \
            if ok else r.fail(f"[{dev}] touch demand-first", detail)


def test_v033_cache8_touch_never_moves_a_frame_between_devices(r):
    """A touched DEMOTED frame stays demoted: `device` still says host RAM, `home` still says the
    GPU, `promotions` stays 0 and the per-device bytes do not move. Its next real `get` then
    promotes it exactly as the unhinted ladder does (the promote row above, unchanged). A hint
    decides which frame a walk takes next; bringing a frame home is left to demand."""
    if "cuda" not in _devices():
        r.skip("touch residency", "no CUDA on this box — nothing to demote")
        return
    with tempfile.TemporaryDirectory() as d:
        src = _frame(res=64, device="cuda")
        c = tex_results.ResultCache(cache_dir=d)
        c.set_vram_budget(0)
        c.put("a", src)
        c.put("b", _frame(res=64, device="cuda", scale=0.5))
        entry = c._ram["a"]

        def residency():
            return (entry.device, entry.home, c.promotions, c.demotions, dict(c._bytes_by_dev))

        was = residency()
        hinted = c.touch("a")
        still = residency()
        got = c.get("a")
        home = (entry.device == entry.home and c.promotions == 1
                and got is not None and got.device.type == "cuda" and torch.equal(got, src))
    ok = (was[0] == "cpu" and was[1].startswith("cuda") and hinted is True and still == was
          and home)
    r.ok("touch: a demoted frame stays demoted through a hint, and its next get promotes it") \
        if ok else r.fail("touch residency", f"before={was} after touch={still} "
                                             f"touch returned {hinted}; get promoted it={home}")


# ── CACHE-11: `touch_promote` — reorder plus promote-if-demoted, nothing else ─────
#
# The ruled narrower grant beside `touch` above: an embedding host that predicts demand wants
# the reorder AND the promotion a `get` hit performs on a demoted frame, but not the hit itself
# (`get`'s hit is the thing it is trying to avoid paying for). Every row below is a rewrite of a
# `touch` row above it, changed only where the grant differs: a demoted frame now comes home.

def test_v033_cache11_touch_promote_matches_touch_when_nothing_is_demoted(r):
    """When there is nothing to promote, `touch_promote` must be `touch` under another name —
    same reorder, same `touches` count, same `False` for an absent/spilled/empty-cache key, and
    NO door reached (not even `_promote`, which has nothing demoted to act on). This is the
    negative half; the positive half (an actually-demoted frame) gets its own CUDA-only row
    below, because a resident-only box can never exercise it."""
    empty_ok = tex_results.ResultCache(
        cache_dir=tempfile.mkdtemp()).touch_promote("anything") is False
    with tempfile.TemporaryDirectory() as d:
        c = tex_results.ResultCache(cache_dir=d, budget_mb=64)
        gone = _frame(res=32, scale=0.25)
        c.put("gone", gone)
        for i, k in enumerate("abc"):
            c.put(k, _frame(res=32, scale=1.0 + i * 0.01))
        c.put("prev", _frame(res=32, scale=0.5), quality=tex_packing.PREVIEW)
        c.put("d", _frame(res=32, scale=0.75))
        spilled = _spill_oldest(c)
        before, t0 = _hint_state(c), c.stats().get("touches")
        with _doors_spied(c) as calls:
            got = {k: c.touch_promote(k) for k in ("a", "prev", "gone", "nobody")}
        after, t1 = _hint_state(c), c.stats().get("touches")
        order = list(c._ram)
        back = c.get("gone")                  # the spill tier is untouched by the hint
        served = back is not None and torch.equal(back, gone) and c.restores == 1
    changed = _changed(before, after)
    ok = (empty_ok and spilled and served and not calls and not changed
          and got == {"a": True, "prev": True, "gone": False, "nobody": False}
          and (t0, t1) == (0, 2) and order == ["b", "c", "a", "prev", "d"])
    r.ok("touch_promote: with nothing demoted it is touch — same reorder, same counts, "
         "no door reached, spilled and absent keys untouched") \
        if ok else r.fail("touch_promote non-demoted parity",
                          f"returned={got} doors={calls} changed={changed} "
                          f"touches {t0}->{t1} order={order} empty_ok={empty_ok} "
                          f"spilled={spilled} served={served}")


def test_v033_cache11_touch_promote_brings_a_demoted_frame_home(r):
    """The grant itself: a demoted frame's NEXT hint — not its next read — brings it home. Same
    residency picture as `test_v033_cache8_touch_never_moves_a_frame_between_devices`, except
    `promotions` moves on the hint instead of waiting for a `get`, `hits` does not move at all
    (the whole reason the host wants this instead of a `get`), and a plain read afterward finds
    the frame already on its home device — no second promotion, no second H2D."""
    if "cuda" not in _devices():
        r.skip("touch_promote residency", "no CUDA on this box — nothing to demote")
        return
    with tempfile.TemporaryDirectory() as d:
        src = _frame(res=64, device="cuda")
        c = tex_results.ResultCache(cache_dir=d)
        c.set_vram_budget(0)
        c.put("a", src)
        c.put("b", _frame(res=64, device="cuda", scale=0.5))
        entry = c._ram["a"]

        def residency():
            return (entry.device, entry.home, c.promotions, c.hits, dict(c._bytes_by_dev))

        was = residency()
        hinted = c.touch_promote("a")
        home_now = residency()
        got = c.get("a")                       # a plain read afterward: no second promotion
        after_get = residency()
    ok = (was[0] == "cpu" and was[1].startswith("cuda") and hinted is True
          and home_now[0] == home_now[1] and home_now[2] == was[2] + 1 and home_now[3] == was[3]
          and after_get[2] == home_now[2] and after_get[3] == was[3] + 1
          and got is not None and got.device.type == "cuda" and torch.equal(got, src))
    r.ok("touch_promote: a demoted frame comes home on the hint, counted in `promotions` and "
         "never in `hits`; the next real read finds it home already and costs no promotion") \
        if ok else r.fail("touch_promote residency",
                           f"before={was} after hint={home_now} touch_promote={hinted} "
                           f"after get={after_get}")


def test_v033_cache11_touch_promote_a_spilled_only_frame_still_does_nothing(r):
    """A frame that is ONLY on disk is not in `_ram` at all, so it is exactly as absent to
    `touch_promote` as to `touch`: no stat, no restore, no promotion attempted — `_promote`
    isn't even a candidate, because there is no RAM entry to hand it. The frame is still
    reachable through an ordinary `get` afterward, proving the hint left the spill tier alone."""
    with tempfile.TemporaryDirectory() as d:
        c = tex_results.ResultCache(cache_dir=d, budget_mb=64)
        spilled_frame = _frame(res=32, scale=0.75)
        c.put("s", spilled_frame)
        for i, k in enumerate("abc"):
            c.put(k, _frame(res=32, scale=1.0 + i * 0.01))
        assert _spill_oldest(c) and "s" not in c._ram          # sanity: genuinely spilled-only
        with _doors_spied(c) as calls:
            hinted = c.touch_promote("s")
        served = c.get("s")
    ok = (hinted is False and not calls
          and served is not None and torch.equal(served, spilled_frame) and c.restores == 1)
    r.ok("touch_promote: a spilled-only frame reaches no door and stays exactly as absent as "
         "it is to `touch`") if ok else \
        r.fail("touch_promote spilled-only", f"hinted={hinted} doors={calls} served={served}")


def test_v033_cache8_a_touched_frame_serves_bit_exact(r):
    """CACHE-2's differential oracle (`test_cache2_hit_is_bit_exact`) run through the hints: a
    cooked frame and a PREVIEW-stored twin of it are touched and probed around a stream of other
    puts — on CUDA under a zero VRAM ceiling, so the residency ladder demotes around them the
    whole time — and then served. Both serve bit-exact to an identical cache that was never
    touched, and the full-precision one bit-exact to the cook itself. A hint cooks nothing, so
    invariant #2 has no row here."""
    for dev in _devices():
        res = tex_engine.cook("@OUT = vec4(@A.rgb * 1.3 + 0.1, 1.0);",
                              {"A": torch.rand(1, 16, 16, 4)}, device_mode=dev,
                              want_lineage=True)
        key, frame = res.lineage["OUT"], res.outputs["OUT"]
        served, probes = [], 0
        for hint in (True, False):
            with tempfile.TemporaryDirectory() as d:
                c = tex_results.ResultCache(cache_dir=d, budget_mb=64)
                if dev == "cuda":
                    c.set_vram_budget(0)
                c.put(key, frame, canvas=(16, 16))
                c.put("prev", frame, quality=tex_packing.PREVIEW)
                for i in range(6):
                    c.put(f"other{i}", _frame(res=16, device=dev, scale=1.0 + i * 0.1))
                    if hint:
                        c.touch(key)
                        c.touch("prev")
                        probes += (key in c) + ("prev" in c)
                served.append((c.get(key), c.get("prev")))
        (h_full, h_prev), (u_full, u_prev) = served
        ok = (probes == 12 and all(t is not None for t in (h_full, h_prev, u_full, u_prev))
              and torch.equal(h_full, frame) and torch.equal(h_full, u_full)
              and torch.equal(h_prev, u_prev) and h_prev.dtype == frame.dtype)
        r.ok(f"[{dev}] touch: a touched frame and its preview twin serve bit-exact to an "
             f"untouched cache, and to the cook") if ok else \
            r.fail(f"[{dev}] touch pixels",
                   f"probes={probes} served={[None if t is None else (t.dtype, str(t.device)) for t in (h_full, h_prev, u_full, u_prev)]}")


def test_v033_cache8_membership_is_a_pure_question(r):
    """`key in cache` answers one question — is the frame RESIDENT in the RAM tier — and changes
    nothing: not the order, not a counter (`touches` included), not a device, and it reaches no
    read, residency or spill door. RAM tier ONLY: a spilled frame reads False while `get` still
    restores it, because disk membership is legitimately unknown at times and the question must
    not go looking (no restore, no directory walk). On CUDA a DEMOTED frame is resident — host
    RAM is part of the RAM tier — and asking about it does not promote it."""
    with tempfile.TemporaryDirectory() as d:
        c = tex_results.ResultCache(cache_dir=d, budget_mb=64)
        gone = _frame(res=32, scale=0.25)
        c.put("gone", gone)
        for i, k in enumerate("abc"):
            c.put(k, _frame(res=32, scale=1.0 + i * 0.01))
        spilled = _spill_oldest(c)
        before, order0, t0 = _hint_state(c), list(c._ram), c.stats().get("touches")
        with _doors_spied(c) as calls:
            got = {k: (k in c) for k in ("a", "c", "gone", "nobody")}
        after, order1, t1 = _hint_state(c), list(c._ram), c.stats().get("touches")
        back = c.get("gone")
        served = back is not None and torch.equal(back, gone) and c.restores == 1
    changed = _changed(before, after)
    ok = (spilled and served and not calls and not changed and order0 == order1
          and t0 == t1 == 0 and all(type(v) is bool for v in got.values())
          and got == {"a": True, "c": True, "gone": False, "nobody": False})
    r.ok("in: resident True, absent and spilled-only False; no order, counter or door moved, "
         "and the spilled frame still restores") if ok else \
        r.fail("in non-effects", f"answers={got} doors={calls} changed={changed} "
                                 f"order {order0}->{order1} touches {t0}->{t1} "
                                 f"spilled={spilled} served={served}")

    if "cuda" not in _devices():
        r.skip("in on a demoted frame", "no CUDA on this box — nothing to demote")
        return
    with tempfile.TemporaryDirectory() as d:
        c = tex_results.ResultCache(cache_dir=d)
        c.set_vram_budget(0)
        c.put("a", _frame(res=64, device="cuda"))
        c.put("b", _frame(res=64, device="cuda", scale=0.5))
        before, order0 = _hint_state(c), list(c._ram)
        with _doors_spied(c) as calls:
            resident = "a" in c
        after, order1 = _hint_state(c), list(c._ram)
        entry = c._ram["a"]
        changed = _changed(before, after)
        ok = (resident is True and entry.device == "cpu" and entry.home.startswith("cuda")
              and c.promotions == 0 and not calls and not changed and order0 == order1)
    r.ok("in: a demoted frame is resident, and asking does not promote it") if ok else \
        r.fail("in on a demoted frame", f"resident={resident} device={entry.device} "
                                        f"home={entry.home} promotions={c.promotions} "
                                        f"doors={calls} changed={changed}")


def _recording_pool(reg, handed, name, holds, order):
    """A governor pool that reports `holds` bytes, records the shortfall `arbitrate` hands it,
    and frees up to what it holds — a stand-in whose only job is to be observed."""
    def evict(dev_type, need, playhead):
        handed[name] = need
        return min(need, holds)
    reg.register(name, lambda dev_type: holds, evict, evict_order=order)


def test_v033_cache8_touch_changes_no_pool_share_under_the_governor(r):
    """The CACHE-5 path end to end — `register_result_cache`, then `arbitrate` — with the pools
    either side of the frame cache replaced by recording stand-ins at their real ladder
    positions (`stdlib` at 10, `graphs` at 90), so the shortfall each is HANDED is observable
    without tearing down a real CUDA graph. Each case runs touched and untouched.

    When the frame pool can cover what reaches it, the touched frame survives and an untouched
    one goes instead. When it cannot, the touch changes nothing the ladder can see: the frame
    pool frees the same bytes (its one-entry floor is still the frame put last, so the eligible
    set did not change) and the pool after it is handed exactly the same shortfall. That is the
    MEM-1 half of the contract: a hint must never make the frame pool under-deliver, because the
    next pool's evictor is the all-or-nothing `free_graphs_only()`. Advice, never a pin."""
    tex_memory._reset_profile_for_test()
    saved = tex_memory._registry
    seen: dict = {}
    nb = 0
    try:
        for case, short_frames in (("covered", 1), ("short", 6)):
            for hinted in (False, True):
                with tempfile.TemporaryDirectory() as d:
                    c = tex_results.ResultCache(cache_dir=d, budget_mb=64)
                    for i, k in enumerate("abcd"):
                        c.put(k, _frame(res=32, scale=1.0 + i * 0.01))
                    nb = c._ram["a"].nbytes
                    reg, handed = tex_memory.CacheRegistry(), {}
                    _recording_pool(reg, handed, "stdlib", nb // 2, 10)
                    _recording_pool(reg, handed, "graphs", 16 * nb, 90)
                    tex_memory._registry = reg
                    tex_memory.register_result_cache(c, name="results")
                    if hinted:
                        c.touch("a")
                    budget = reg.total_bytes("cpu") - (nb // 2 + short_frames * nb)
                    freed = reg.arbitrate("cpu", budget=budget)
                    seen[case, hinted] = (freed, dict(handed), [k for k in "abcd" if k in c._ram])
    finally:
        tex_memory._registry = saved
        tex_memory._reset_profile_for_test()
    covered = (seen["covered", False][2] == ["b", "c", "d"]
               and seen["covered", True][2] == ["a", "c", "d"]
               and seen["covered", False][:2] == seen["covered", True][:2]
               and "graphs" not in seen["covered", True][1])
    short = (seen["short", False] == seen["short", True]
             and seen["short", True][2] == ["d"]
             and seen["short", True][1].get("graphs") == 3 * nb)
    r.ok("touch: under arbitrate a touched frame survives when the frame pool can cover the "
         "need, and when it cannot every pool frees and is handed exactly what it was untouched") \
        if covered and short else r.fail("touch governor", f"{seen}")


def test_v033_cache8_touch_and_in_survive_a_threaded_race(r):
    """The concurrency contract the class states — thread-safe since CACHE-7, the lock covering
    structure and byte accounting and never I/O — with both hint doors raced against every other
    door into the table: two writers `put` over a budget that holds three frames (so eviction
    and the disk spill race too), a reader `get`s (restoring, and on CUDA promoting), a governor
    thread calls `evict_bytes`, and a hinter and a prober call `touch` and `in` throughout. On
    CUDA the VRAM ceiling is 0, so every put demotes and every hit promotes.

    Checked once every thread has stopped, where the answers are exact rather than racy: nothing
    raised or hung; every hit served its own key's pixels; `hits + misses` equals the gets issued
    and `touches` equals the touches that found a resident key — no hint leaked into the read
    counters under contention and no increment was lost; the per-device byte buckets equal a
    recount of the entries; both drain queues and the in-flight set are empty; every home is the
    cook device and every frame is at home or demoted; `in` agrees with the table for every key;
    and a touch still ranks just below the most recent put."""
    for dev in _devices():
        with tempfile.TemporaryDirectory() as d:
            c = tex_results.ResultCache(cache_dir=d, budget_mb=0.05)  # three 32² x4 fp32 frames
            if dev == "cuda":
                c.set_vram_budget(0)
            keys = [f"k{i}" for i in range(12)]
            frames = [_frame(res=32, device=dev, scale=1.0 + i * 0.01) for i in range(12)]
            nb = frames[0].numel() * frames[0].element_size()
            bucket = tex_results._dev_bucket(dev)
            stop, errors = threading.Event(), []
            # Fixed keys, each written by exactly one thread: no lost update between them.
            n = {"put0": 0, "put1": 0, "get": 0, "wrong": 0, "evict": 0, "touch": 0,
                 "resident": 0, "in": 0}

            def writer(slot):
                def step(i):
                    j = (5 * i + 7 * slot) % len(keys)
                    c.put(keys[j], frames[j])
                    n[f"put{slot}"] += 1
                return step

            def reader(i):
                j = (7 * i) % len(keys)
                got = c.get(keys[j], copy=bool(i & 1))
                n["get"] += 1
                if got is not None and not torch.equal(got.to(frames[j].device), frames[j]):
                    n["wrong"] += 1

            def governor(i):
                c.evict_bytes(nb, dev_type=bucket)
                n["evict"] += 1

            def hinter(i):
                if c.touch(keys[(3 * i) % len(keys)]):
                    n["resident"] += 1
                n["touch"] += 1

            def prober(i):
                answer = keys[(11 * i) % len(keys)] in c
                if type(answer) is not bool:
                    raise TypeError(f"`in` answered {answer!r}")
                n["in"] += 1

            def looped(name, step):
                def run():
                    i = 0
                    try:
                        while not stop.is_set():
                            step(i)
                            i += 1
                    except Exception as exc:              # noqa: BLE001 — that IS the finding
                        errors.append(f"{name}: {type(exc).__name__}: {exc}")
                return threading.Thread(target=run, name=name, daemon=True)

            threads = [looped("writer0", writer(0)), looped("writer1", writer(1)),
                       looped("reader", reader), looped("governor", governor),
                       looped("hinter", hinter), looped("prober", prober)]
            for t in threads:
                t.start()
            time.sleep(0.6)
            stop.set()
            for t in threads:
                t.join(timeout=30.0)
            hung = [t.name for t in threads if t.is_alive()]

            recount, buckets = _recount(c)
            with c._lock:
                idle = (len(c._pending_spills), len(c._pending_demotes), len(c._demoting))
                homes = {e.home for e in c._ram.values()}
                placed = all(e.device in (e.home, "cpu") for e in c._ram.values())
            st = c.stats()
            agree = all((k in c) == (k in c._ram) for k in keys)
            # The order contract, intact after the race: room again, three fresh puts, and a
            # touch of the oldest lands it just below the last one.
            c.set_budget(64)
            for j in range(3):
                c.put(f"after{j}", frames[j])
            c.touch("after0")
            tail = list(c._ram)[-3:]
        progressed = all(n[k] > 0 for k in ("put0", "put1", "get", "evict", "touch", "in"))
        ok = (not errors and not hung and progressed and n["wrong"] == 0
              and st["hits"] + st["misses"] == n["get"] and st["touches"] == n["resident"]
              and recount == buckets and idle == (0, 0, 0) and placed and agree
              and homes <= {str(frames[0].device)} and tail == ["after1", "after0", "after2"])
        r.ok(f"[{dev}] touch/in raced put/get/evict_bytes: {n['touch']} touches, {n['in']} "
             f"probes, {n['get']} gets, {st['evictions']} evictions, {st['demotions']} "
             f"demotions — counters, buckets and order exact") if ok else \
            r.fail(f"[{dev}] touch/in race",
                   f"errors={errors[:3]} hung={hung} ops={n} hits+misses="
                   f"{st['hits'] + st['misses']} touches={st['touches']} recount={recount} "
                   f"buckets={buckets} queues={idle} homes={homes} placed={placed} "
                   f"agree={agree} tail={tail}")


class _ParkedOrder(OrderedDict):
    """The cache's LRU table with one trap in it: the first `move_to_end` of `trap` waits at a
    gate before it moves anything. Everything else is the plain OrderedDict, so the cache runs
    its shipped code over it."""

    def __init__(self, *a, trap=None, **kw):
        super().__init__(*a, **kw)
        self.trap = trap
        self.at_gate, self.resume = threading.Event(), threading.Event()

    def move_to_end(self, key, last=True):
        if key == self.trap:
            self.trap = None
            self.at_gate.set()
            self.resume.wait(20)
        return super().move_to_end(key, last)


def test_v033_cache8_touch_is_atomic_against_a_concurrent_put(r):
    """`touch` reads the most recent entry and then moves two entries. If a `put` could land
    between the read and the moves, the touch would put the STALE top entry back above the frame
    that put just admitted — a hint outranking demand, and on CUDA a just-cooked frame handed to
    the demotion walk. Deterministic: the touch is parked inside its reorder, a put is started on
    another thread and given a full second to land, and only then is the touch let go. With the
    lock held across the read and both moves the put cannot land early, so it arrives on top."""
    with tempfile.TemporaryDirectory() as d:
        c = tex_results.ResultCache(cache_dir=d, budget_mb=64)
        for i, k in enumerate("abcd"):
            c.put(k, _frame(res=32, scale=1.0 + i * 0.01))
        c._ram = _ParkedOrder(c._ram, trap="a")
        toucher = threading.Thread(target=c.touch, args=("a",), daemon=True)
        putter = threading.Thread(target=c.put, args=("e", _frame(res=32, scale=2.0)),
                                  daemon=True)
        toucher.start()
        try:
            parked = c._ram.at_gate.wait(10)
            putter.start()
            putter.join(1.0)                  # a full second to land, if the lock allows it
            landed_early = not putter.is_alive()
        finally:
            c._ram.resume.set()
            toucher.join(20)
            putter.join(20)
        order = list(c._ram)
    ok = parked and not landed_early and order == ["b", "c", "a", "d", "e"]
    r.ok("touch: reading the top entry and both moves are one critical section — a racing put "
         "still lands on top") if ok else \
        r.fail("touch atomicity", f"parked={parked} put landed mid-touch={landed_early} "
                                  f"order={order}")


def test_v033_cache8_hints_never_wait_on_or_undo_in_flight_work(r):
    """The lock rule from the hint side. A spill's disk write and a demotion's D2H both run
    OUTSIDE the lock with their victim in a half-way state: a spill victim is already out of the
    table, a demotion victim is still in it (on the GPU, marked in flight). Each is parked at its
    copy while both hints are asked about the victim from another thread. They answer at once —
    neither waits on the I/O — and they answer the truth about each half-way state: the spill
    victim is not resident (False, and no restore is attempted), the demotion victim is (True,
    and the touch reorders it). Released, the parked work finishes exactly as it would have: the
    spilled frame restores bit-exact, and the demotion commits — a touch does not rescue a victim
    a walk has already chosen, and moves neither `device` nor `home`."""
    from TEX_Wrangle.tex_runtime import streams
    real_egress = streams.egress

    def gate_first_egress():
        at_gate, resume, armed = threading.Event(), threading.Event(), [True]

        def gated(src, **kw):
            if armed[0]:
                armed[0] = False
                at_gate.set()
                resume.wait(20)
            return real_egress(src, **kw)
        return gated, at_gate, resume

    def ask(c, key):
        """Both hints about `key`, from a helper thread: (answered within 5 s, touch, in)."""
        out: dict = {}
        t = threading.Thread(target=lambda: out.update(touch=c.touch(key), resident=key in c),
                             daemon=True)
        t.start()
        t.join(5.0)
        return not t.is_alive(), out.get("touch"), out.get("resident")

    # CPU: an eviction's disk write, parked mid-copy.
    with tempfile.TemporaryDirectory() as d:
        c = tex_results.ResultCache(cache_dir=d, budget_mb=64)
        victim = _frame(res=32, scale=0.25)
        c.put("v", victim)
        for i, k in enumerate("abc"):
            c.put(k, _frame(res=32, scale=1.0 + i * 0.01))
        gated, at_gate, resume = gate_first_egress()
        spiller = threading.Thread(target=c.evict_bytes, args=(c._ram["v"].nbytes,),
                                   kwargs={"dev_type": "cpu"}, daemon=True)
        streams.egress = gated
        try:
            spiller.start()
            parked = at_gate.wait(10)
            with _doors_spied(c) as calls:
                answered, touched, resident = ask(c, "v")
            midway = spiller.is_alive()
        finally:
            resume.set()
            spiller.join(20)
            streams.egress = real_egress
        back = c.get("v")
        cpu = (parked and answered and midway and touched is False and resident is False
               and not calls and back is not None and torch.equal(back, victim)
               and c.spills == 1 and c.restores == 1)
    r.ok("hints: asked mid-spill they answer at once, not resident, and the frame still "
         "restores bit-exact") if cpu else \
        r.fail("hints during an in-flight spill",
               f"parked={parked} answered={answered} still-parked={midway} touch={touched} "
               f"in={resident} doors={calls} spills={c.spills} restores={c.restores} "
               f"restored={back is not None}")

    if "cuda" not in _devices():
        r.skip("hints during an in-flight demotion", "no CUDA on this box — nothing to demote")
        return
    # CUDA: a demotion's D2H, parked mid-copy with its victim still in the table.
    with tempfile.TemporaryDirectory() as d:
        c = tex_results.ResultCache(cache_dir=d, budget_mb=64)
        for i, k in enumerate("abc"):
            c.put(k, _frame(res=64, device="cuda", scale=1.0 + i * 0.01))
        gated, at_gate, resume = gate_first_egress()
        demoter = threading.Thread(target=c.set_vram_budget, args=(0,), daemon=True)
        streams.egress = gated
        try:
            demoter.start()
            parked = at_gate.wait(10)
            with c._lock:
                in_flight = set(c._demoting)
            with _doors_spied(c) as calls:
                answered, touched, resident = ask(c, "a")
            midway = demoter.is_alive()
            order = list(c._ram)
            on_gpu = c._ram["a"].device.startswith("cuda")
        finally:
            resume.set()
            demoter.join(20)
            streams.egress = real_egress
        entry = c._ram["a"]
        recount, buckets = _recount(c)
        gpu = (parked and answered and midway and in_flight == {"a"} and touched is True
               and resident is True and not calls and on_gpu and order == ["b", "a", "c"]
               and entry.device == "cpu" and entry.home.startswith("cuda")
               and c._ram["b"].device == "cpu" and c.demotions == 2 and c.promotions == 0
               and recount == buckets and not c._demoting)
    r.ok("hints: asked mid-demotion they answer at once, resident, and the demotion still "
         "commits with its home intact") if gpu else \
        r.fail("hints during an in-flight demotion",
               f"parked={parked} answered={answered} still-parked={midway} "
               f"in_flight={in_flight} touch={touched} in={resident} doors={calls} "
               f"order={order} on_gpu_mid={on_gpu} after=({entry.device}, {entry.home}) "
               f"demotions={c.demotions} promotions={c.promotions} "
               f"recount={recount} buckets={buckets}")
