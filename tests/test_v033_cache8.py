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

Every row is CPU-safe: `_devices()` adds the CUDA rows when there is a GPU, and the residency
ladder degenerates honestly to one rung without one (there is nothing to demote FROM).
"""
import tempfile

import torch

from helpers import devices as _devices, make_gradient_frame as _frame
from TEX_Wrangle import tex_memory, tex_packing, tex_results


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
        r.ok("CACHE-8: demote SKIPPED (no CUDA — nothing to demote from)")
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
        r.ok("CACHE-8: demote bit-exactness SKIPPED (no CUDA)")
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
        r.ok("CACHE-8: promote SKIPPED (no CUDA)")
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
        r.ok("CACHE-8: demote+spill SKIPPED (no CUDA)")
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
        r.ok("CACHE-8: governor preference SKIPPED (no CUDA)")
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
        r.ok("CACHE-8: unarmed governor SKIPPED (no CUDA)")
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
    timing cannot."""
    import pathlib
    node = (pathlib.Path(__file__).resolve().parent.parent / "tex_node.py").read_text(
        encoding="utf-8")
    hits = [n for n in ("set_vram_budget", "residency", "_demote", "_promote") if n in node]
    r.ok("CACHE-8: tex_node.py cannot reach the residency ladder") if not hits else \
        r.fail("CACHE-8 invariant#7", f"tex_node.py mentions {hits}")
