"""BRIEF-10 — integrity BEFORE deserialise in the on-disk pickle caches.

`tex_cache` and `tex_results` reload compiled programs and spilled frames with `pickle.load()`,
whose `__reduce__` executes DURING the load — before any stored field is inspected. The cache
dir is writable by more than this process by design (an embedding host may run a second instance
sharing it), so a crafted file dropped there by another principal is code execution on the next
cook that hits its fingerprint.

The fix authenticates the bytes with a keyed MAC (the key lives outside the cache dir) BEFORE
any deserialise, at all three sites. These rows pin it RED-FIRST:

  * CRAFTED — a pickle whose `__reduce__` would run (a harmless marker-file side effect) is never
    deserialised at `_load_from_disk`, `_load_codegen_from_disk`, or `_restore`. On the base tree
    the marker appears (the load ran it); on the fixed tree it never does.
  * WARM HIT — a legitimately written entry still loads, so the gate does not over-reject its own
    files (compile `.pkl`/`.cg`, and a spilled `.frame`).
  * MIGRATION — an unsigned (old-format / foreign) entry is a silent MISS that recomputes or
    re-cooks, never an error and never served. On the base tree it is a HIT.
"""
from helpers import *

import pickle as _pickle

import torch

from TEX_Wrangle.tex_cache import TEXCache
from TEX_Wrangle import tex_results
from TEX_Wrangle.tex_runtime import compiled as _C


class _ExecOnLoad:
    """Its `__reduce__` makes unpickling call `open(marker, "w")` — a harmless, pure-stdlib side
    effect that PROVES the loader deserialised attacker bytes. The class itself is never stored;
    pickle serialises the reduce output `(open, (marker, "w"))`, so `open` is resolved by
    reference at load time whether or not this class is importable."""

    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (open, (self.marker, "w"))


def _crafted(marker_path) -> bytes:
    return _pickle.dumps(_ExecOnLoad(marker_path), protocol=_pickle.HIGHEST_PROTOCOL)


def _forged(marker_path) -> bytes:
    """A crafted pickle carrying a well-formed trailer whose MAC tag is wrong (32 zero bytes).
    The magic/format gate passes it; only the MAC compare refuses it — so a row using this FAILS
    if `hmac.compare_digest` is deleted or forced always-true (the F4 mutation-proof)."""
    return _crafted(marker_path) + b"TEXm1" + b"\x00" * 32


_CODE = "@OUT = @A * 0.7 + 0.1;"
_CG_CODE = ("vec3 acc = vec3(0.0);\n"
            "for (int i = 0; i < 4; i = i + 1) { acc = acc + @A.rgb * 0.1; }\n"
            "@OUT = vec4(acc, 1.0);\n")


def _frame(res=32):
    return torch.rand(1, res, res, 4)


# ── CRAFTED: __reduce__ never runs at any of the three sites ───────────────────

def test_brief10_pkl_site_never_executes_crafted_reduce(r: SubTestResult):
    print("\n--- BRIEF-10: _load_from_disk authenticates before pickle.load ---")
    try:
        d = Path(tempfile.mkdtemp())
        marker = d / "PWNED_pkl"
        cache = TEXCache(cache_dir=d)
        bt = {"A": TEXType.VEC4}
        fp = cache.fingerprint(_CODE, bt)
        cache._disk_path(fp).write_bytes(_crafted(str(marker)))
        got = cache._load_from_disk(fp, bt)
        ok = (not marker.exists()) and got is None
        r.ok("a crafted .pkl is a miss; its __reduce__ never runs") if ok else \
            r.fail("BRIEF-10 .pkl crafted",
                   f"marker_created={marker.exists()} load_returned_non_none={got is not None}")
        shutil.rmtree(d, ignore_errors=True)
    except Exception as e:
        r.fail("BRIEF-10 .pkl crafted", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


def test_brief10_cg_site_never_executes_crafted_reduce(r: SubTestResult):
    print("\n--- BRIEF-10: _load_codegen_from_disk authenticates before pickle.load ---")
    try:
        d = Path(tempfile.mkdtemp())
        marker = d / "PWNED_cg"
        cache = TEXCache(cache_dir=d)
        fp = "deadbeef" * 8
        cache._cg_path(fp).write_bytes(_crafted(str(marker)))
        got = cache._load_codegen_from_disk(fp)
        ok = (not marker.exists()) and got is None
        r.ok("a crafted .cg is a miss; its __reduce__ never runs") if ok else \
            r.fail("BRIEF-10 .cg crafted",
                   f"marker_created={marker.exists()} load_returned_non_none={got is not None}")
        shutil.rmtree(d, ignore_errors=True)
    except Exception as e:
        r.fail("BRIEF-10 .cg crafted", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


def test_brief10_frame_site_never_executes_crafted_reduce(r: SubTestResult):
    print("\n--- BRIEF-10: _restore authenticates before pickle.load ---")
    try:
        d = Path(tempfile.mkdtemp())
        marker = d / "PWNED_frame"
        c = tex_results.ResultCache(cache_dir=str(d), budget_mb=0)
        key = "k"
        Path(c._disk_path(key)).write_bytes(_crafted(str(marker)))
        with c._lock:
            c._spilled = None                     # unknown → _restore stats and finds the file
        frame, orig, _fence = c._restore(key)
        ok = (not marker.exists()) and frame is None
        r.ok("a crafted .frame is a miss; its __reduce__ never runs") if ok else \
            r.fail("BRIEF-10 .frame crafted",
                   f"marker_created={marker.exists()} restore_returned_non_none={frame is not None}")
        shutil.rmtree(d, ignore_errors=True)
    except Exception as e:
        r.fail("BRIEF-10 .frame crafted", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


# ── FORGED TRAILER: the MAC compare itself is load-bearing (F4 mutation-proof) ─

def test_brief10_pkl_site_rejects_a_forged_trailer(r: SubTestResult):
    print("\n--- BRIEF-10: a forged .pkl trailer (bad tag) is refused by the MAC ---")
    try:
        d = Path(tempfile.mkdtemp())
        marker = d / "FORGED_pkl"
        cache = TEXCache(cache_dir=d)
        bt = {"A": TEXType.VEC4}
        fp = cache.fingerprint(_CODE, bt)
        cache._disk_path(fp).write_bytes(_forged(str(marker)))
        got = cache._load_from_disk(fp, bt)
        ok = (not marker.exists()) and got is None
        r.ok("a forged-tag .pkl is refused; __reduce__ never runs (MAC compare is live)") if ok \
            else r.fail("BRIEF-10 .pkl forged",
                        f"marker_created={marker.exists()} load_non_none={got is not None}")
        shutil.rmtree(d, ignore_errors=True)
    except Exception as e:
        r.fail("BRIEF-10 .pkl forged", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


def test_brief10_cg_site_rejects_a_forged_trailer(r: SubTestResult):
    print("\n--- BRIEF-10: a forged .cg trailer (bad tag) is refused by the MAC ---")
    try:
        d = Path(tempfile.mkdtemp())
        marker = d / "FORGED_cg"
        cache = TEXCache(cache_dir=d)
        fp = "feedface" * 8
        cache._cg_path(fp).write_bytes(_forged(str(marker)))
        got = cache._load_codegen_from_disk(fp)
        ok = (not marker.exists()) and got is None
        r.ok("a forged-tag .cg is refused; __reduce__ never runs (MAC compare is live)") if ok \
            else r.fail("BRIEF-10 .cg forged",
                        f"marker_created={marker.exists()} load_non_none={got is not None}")
        shutil.rmtree(d, ignore_errors=True)
    except Exception as e:
        r.fail("BRIEF-10 .cg forged", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


def test_brief10_frame_site_rejects_a_forged_trailer(r: SubTestResult):
    print("\n--- BRIEF-10: a forged .frame trailer (bad tag) is refused by the MAC ---")
    try:
        d = Path(tempfile.mkdtemp())
        marker = d / "FORGED_frame"
        c = tex_results.ResultCache(cache_dir=str(d), budget_mb=0)
        Path(c._disk_path("k")).write_bytes(_forged(str(marker)))
        with c._lock:
            c._spilled = None
        frame, _orig, _fence = c._restore("k")
        ok = (not marker.exists()) and frame is None
        r.ok("a forged-tag .frame is refused; __reduce__ never runs (MAC compare is live)") if ok \
            else r.fail("BRIEF-10 .frame forged",
                        f"marker_created={marker.exists()} restore_non_none={frame is not None}")
        shutil.rmtree(d, ignore_errors=True)
    except Exception as e:
        r.fail("BRIEF-10 .frame forged", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


# ── WARM HIT: a legitimately written entry still loads ─────────────────────────

def test_brief10_signed_entries_still_load(r: SubTestResult):
    print("\n--- BRIEF-10: signed .pkl/.cg/.frame still load (no over-rejection) ---")
    try:
        d = Path(tempfile.mkdtemp())
        cache = TEXCache(cache_dir=d)
        bt = {"A": TEXType.VEC4}
        prog, tm, *_ = cache.compile_tex(_CODE, bt)     # writes a signed .pkl
        cache.clear_memory()
        hit = cache.get(_CODE, bt)
        pkl_ok = hit is not None

        # The codegen sidecar is minted by the singleton cache `_get_or_make_codegen_fn` reaches
        # (TEX_CACHE_DIR-backed in the suite), so drive the .cg leg through it — mirrors v015.
        from TEX_Wrangle.tex_cache import get_cache
        gc = get_cache()
        bt2 = {"A": TEXType.VEC3}
        prog2, tm2, _r, _a, _p, used2 = gc.compile_tex(_CG_CODE, bt2)
        fp2 = gc.fingerprint(_CG_CODE, bt2)
        gc._codegen_memory.pop(fp2, None)
        gc._cg_path(fp2).unlink(missing_ok=True)
        fn1 = _C._get_or_make_codegen_fn(prog2, tm2, fp2)   # writes a signed .cg
        gc._codegen_memory.pop(fp2, None)                   # simulate a restart: drop RAM tier
        cg_ok = (fn1 is not None and gc._cg_path(fp2).exists()
                 and gc.get_codegen_fn(fp2) is not None)    # materialised from the signed sidecar

        c = tex_results.ResultCache(cache_dir=str(d), budget_mb=0)
        f = _frame()
        c.put("frame", f)
        c.put("evictor", f)                             # forces "frame" out to a signed .frame
        got = c.get("frame")
        frame_ok = got is not None and float((got - f).abs().max()) < 1e-6

        ok = pkl_ok and cg_ok and frame_ok
        r.ok("signed .pkl, .cg and .frame all round-trip") if ok else \
            r.fail("BRIEF-10 warm hit",
                   f"pkl_hit={pkl_ok} cg_hit={cg_ok} frame_hit={frame_ok}")
        shutil.rmtree(d, ignore_errors=True)
    except Exception as e:
        r.fail("BRIEF-10 warm hit", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


# ── MIGRATION: an unsigned entry is a silent miss that recomputes ──────────────

def test_brief10_unsigned_pkl_is_a_silent_miss_then_recompiles(r: SubTestResult):
    print("\n--- BRIEF-10: an unsigned .pkl is a miss that recompiles (not served) ---")
    try:
        d = Path(tempfile.mkdtemp())
        cache = TEXCache(cache_dir=d)
        bt = {"A": TEXType.VEC4}
        cache.compile_tex(_CODE, bt)
        fp = cache.fingerprint(_CODE, bt)
        # An unsigned but otherwise-valid record (version matches → a HIT on the base tree).
        import TEX_Wrangle.tex_cache as _TC
        prog, *_ = cache.compile_tex(_CODE, bt)
        with open(cache._disk_path(fp), "wb") as fh:
            _pickle.dump({"version": _TC._AST_EPOCH, "program": prog,
                          "binding_types": {k: v.value for k, v in bt.items()},
                          "timestamp": 0.0}, fh, protocol=_pickle.HIGHEST_PROTOCOL)
        cache.clear_memory()
        miss = cache._load_from_disk(fp, bt)            # HEAD: None (unsigned); base: not None
        # ...and the engine still recomputes and re-signs, silently.
        cache.clear_memory()
        recompiled = cache.compile_tex(_CODE, bt)
        raw = open(cache._disk_path(fp), "rb").read()   # literal magic → no head-only import
        resigned = len(raw) > 37 and raw[-37:-32] == b"TEXm1"
        ok = miss is None and recompiled is not None and resigned
        r.ok("unsigned .pkl → silent miss → recompiled + re-signed") if ok else \
            r.fail("BRIEF-10 unsigned .pkl",
                   f"miss={miss is None} recompiled={recompiled is not None} resigned={resigned}")
        shutil.rmtree(d, ignore_errors=True)
    except Exception as e:
        r.fail("BRIEF-10 unsigned .pkl", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


def test_brief10_unsigned_frame_is_a_silent_miss(r: SubTestResult):
    print("\n--- BRIEF-10: an unsigned .frame is a miss (not served) ---")
    try:
        d = Path(tempfile.mkdtemp())
        c = tex_results.ResultCache(cache_dir=str(d), budget_mb=0)
        f = _frame()
        # A fully valid v2 record, written UNSIGNED — a HIT on the base tree, a miss on the fix.
        rec = {"t": f[0].clone(), "fmt": tex_results._FRAME_FORMAT, "device": "cpu",
               "canvas": None, "epoch": tex_results.env_epoch(), "orig": None,
               "viewed": None, "quality": None}
        with open(c._disk_path("legacy"), "wb") as fh:
            _pickle.dump(rec, fh, protocol=_pickle.HIGHEST_PROTOCOL)
        with c._lock:
            c._spilled = None
        served = c.get("legacy")
        # A genuinely signed put of the same key then warm-hits (re-cook path is honest).
        c.put("legacy", f)
        c.put("evictor", f)
        rehit = c.get("legacy")
        ok = served is None and rehit is not None and float((rehit - f).abs().max()) < 1e-6
        r.ok("unsigned .frame → miss; a re-cooked signed frame warm-hits") if ok else \
            r.fail("BRIEF-10 unsigned .frame",
                   f"unsigned_served={served is not None} resigned_hit={rehit is not None}")
        shutil.rmtree(d, ignore_errors=True)
    except Exception as e:
        r.fail("BRIEF-10 unsigned .frame", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


# ── N1: malformed-key repair must not delete a peer's freshly published key ────

def test_brief10_key_repair_spares_a_peers_republished_key(r: SubTestResult):
    """N1 (re-review): repairing a malformed key removed whatever was at the path — including a
    peer's good key published between the probe and the remove — splitting racers onto different
    keys. The fix removes only a file whose stat still matches the malformed one probed. This
    forces the ordering deterministically: a peer republishes the good key K_B during the guard
    `os.stat`, and the repair must then LEAVE it and converge to it."""
    import TEX_Wrangle.tex_recovery as R
    try:
        home = Path(tempfile.mkdtemp())
        keypath = home / R._MAC_KEY_FILE
        keypath.write_bytes(b"\x01\x02\x03")            # a malformed 3-byte key
        K_B = b"B" * R._MAC_KEY_LEN                      # the peer's good, published key
        real_stat = R.os.stat
        state = {"peer_published": False}

        def stat_letting_a_peer_in(p, *a, **k):
            try:
                same = os.path.abspath(p) == os.path.abspath(str(keypath))
            except TypeError:
                same = False                            # a stat by fd, not our path
            if same and not state["peer_published"]:
                state["peer_published"] = True          # peer republishes between probe and guard
                with open(keypath, "wb") as f:
                    f.write(K_B)
            return real_stat(p, *a, **k)

        orig_home = R._mac_key_home
        R._mac_key_home = lambda: str(home)
        R.os.stat = stat_letting_a_peer_in
        try:
            got = R._resolve_or_create_key()
        finally:
            R.os.stat = real_stat
            R._mac_key_home = orig_home

        on_disk = keypath.read_bytes() if keypath.exists() else None
        ok = state["peer_published"] and got == K_B and on_disk == K_B
        r.ok("N1: a peer's republished key survives repair; the racer converges to it") if ok \
            else r.fail("BRIEF-10 N1 key repair",
                        f"peer_ran={state['peer_published']} returned_is_KB={got == K_B} "
                        f"on_disk_is_KB={on_disk == K_B}")
        shutil.rmtree(home, ignore_errors=True)
    except Exception as e:
        r.fail("BRIEF-10 N1 key repair", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
