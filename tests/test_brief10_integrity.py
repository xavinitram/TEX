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


# ── RESTORE-462: a TRANSIENT open() failure must never be treated as "tampered" ─
#
# `load_verified` reads `path` with one `open(...).read()`. On the base tree, ANY `OSError` from
# that open/read — a Windows sharing violation from a real-time scanner or indexer holding a
# transient handle, a momentary EMFILE, a flaky network/cloud-synced cache dir, anything at all —
# is caught by the SAME `except OSError: return _UNVERIFIED` that also covers "opened fine, but
# the trailer/MAC/pickle is bad". Every caller (`_restore`, `_load_from_disk`,
# `_load_codegen_from_disk`) treats `_UNVERIFIED` as license to delete the file. A transient
# failure to even OPEN the file therefore destroys a perfectly valid, previously-spilled frame
# that a retry moments later would have served — an own spill intermittently rejected on restore,
# across processes, and never before this trailer existed. The module already knows this
# distinction matters — see `_probe_key`'s "unreadable" branch, which never removes a key file it
# could not read — but `load_verified` never got the same treatment.


def test_restore462_transient_open_failure_is_a_miss_not_a_deletion(r: SubTestResult):
    print("\n--- RESTORE-462: a transient open() OSError must not delete a valid .frame ---")
    try:
        d = Path(tempfile.mkdtemp())
        c = tex_results.ResultCache(cache_dir=str(d), budget_mb=0)
        f = _frame()
        c.put("k", f)
        c.put("evictor", f)                      # forces "k" out to a signed, valid .frame
        path = c._disk_path("k")
        if not os.path.exists(path):
            r.fail("RESTORE-462 transient open error", "setup did not spill 'k'")
            shutil.rmtree(d, ignore_errors=True)
            return

        import builtins
        real_open = builtins.open
        state = {"raised": False}
        target = os.path.abspath(str(path))

        def flaky_open(file, *a, **kw):
            if not state["raised"] and os.path.abspath(os.fspath(file)) == target:
                state["raised"] = True
                raise PermissionError(
                    13, "The process cannot access the file because it is being used "
                        "by another process")
            return real_open(file, *a, **kw)

        with c._lock:
            c._spilled = None                    # unknown -> _restore stats and finds the file
        builtins.open = flaky_open
        try:
            frame1, _orig1, _fence1 = c._restore("k")
        finally:
            builtins.open = real_open

        survived = os.path.exists(path)          # THE claim: a transient miss must not delete it
        with c._lock:
            c._spilled = None
        frame2, _orig2, _fence2 = c._restore("k")   # an unobstructed retry must still serve it
        retried_ok = frame2 is not None and float((frame2 - f).abs().max()) < 1e-6

        ok = survived and retried_ok
        r.ok("a transient open() failure is a miss; the frame survives and a retry serves it") \
            if ok else r.fail(
                "RESTORE-462 transient open error",
                f"first_restore_hit={frame1 is not None} file_survived={survived} "
                f"retry_hit={retried_ok}")
        shutil.rmtree(d, ignore_errors=True)
    except Exception as e:
        r.fail("RESTORE-462 transient open error", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


# ── RESTORE-462 §4: the per-user MAC key file itself is read in TEXT mode on Windows ───────────
#
# `_probe_key` reads the key with a raw `os.open(path, os.O_RDONLY)` / `os.read` — the low-level
# fd path every OTHER writer in this module deliberately avoids by minting through
# `tempfile.mkstemp` (which defaults to BINARY mode). With no `O_BINARY`, Windows opens the file
# in TEXT mode: `os.read` stops at the first 0x1A (Ctrl-Z, the legacy text-mode EOF marker) and
# folds every 0x0D 0x0A pair to 0x0A. A 32-byte random key contains a 0x1A byte with probability
# 1-(255/256)**32 ~= 11.8% per mint (a CRLF-shaped byte pair is misread the same way), so roughly
# one key in nine reads back SHORT, `_probe_key` classes it "malformed", `_resolve_or_create_key`
# deletes it and mints a replacement — and every earlier process's signed spill / `.pkl` / `.cg`
# then fails `load_verified` in every LATER process, with an intact trailer and a matching epoch:
# the MAC verdict fires because the KEY changed under it, not because anything was tampered. This
# is the actual cause behind the host's report; `_UNVERIFIED`-vs-`_UNREADABLE` above is a real,
# separate defensive gap it does not explain.


def test_restore462_probe_key_reads_the_key_file_in_binary_mode(r: SubTestResult):
    print("\n--- RESTORE-462: _probe_key must not read the MAC key in text mode ---")
    import TEX_Wrangle.tex_recovery as R
    try:
        home = Path(tempfile.mkdtemp())
        keypath = home / R._MAC_KEY_FILE

        # Two independent byte patterns, each well inside the ~11.8%-per-mint natural rate, not
        # edge cases invented for the test: one trips the text-mode EOF byte, one trips CRLF
        # folding.
        key_ctrlz = bytes([0x1A]) + bytes(range(1, 32))                      # 32 bytes, has 0x1A
        key_crlf = bytes(range(1, 15)) + b"\x0d\x0a" + bytes(range(15, 31))   # 32 bytes, has CRLF

        for name, key in (("0x1A", key_ctrlz), ("CRLF", key_crlf)):
            keypath.write_bytes(key)

            # Test the read helper directly (meaningful on every platform: a persisted 32-byte
            # key must always read back as the SAME 32 bytes).
            kind, info = R._probe_key(str(keypath))
            probe_ok = kind == "ok" and info == key
            r.ok(f"_probe_key reads a {name}-containing key back unchanged") if probe_ok else \
                r.fail(f"RESTORE-462 probe_key {name}", f"kind={kind} info={info!r}")

            # A fresh-process-style reload: drop the memo, point the key home at this dir, and
            # confirm the file comes back UNCHANGED (not replaced by a freshly minted key) and a
            # signature made with it still verifies — the end-to-end shape of the host's report.
            R._mac_key_cache = None
            orig_home = R._mac_key_home
            R._mac_key_home = lambda: str(home)
            try:
                reloaded = R._mac_key()
            finally:
                R._mac_key_home = orig_home
            reload_ok = reloaded == key and keypath.read_bytes() == key

            marker = home / f"sig_{name}"
            signed = R.sign_pickle(str(marker), {"v": name})
            verified = R.load_verified(str(marker)) if signed else None
            sign_ok = signed and isinstance(verified, dict) and verified.get("v") == name

            r.ok(f"a {name}-containing key survives a fresh-process reload and still signs/verifies") \
                if (reload_ok and sign_ok) else r.fail(
                    f"RESTORE-462 key reload {name}",
                    f"probe_ok={probe_ok} reload_ok={reload_ok} sign_ok={sign_ok}")
            R._mac_key_cache = None
            keypath.unlink(missing_ok=True)
            marker.unlink(missing_ok=True)

        # Platform-portable lock, per the brief: assert the probe's os.open flags include
        # O_BINARY WHERE the platform defines one — a real assertion on Windows, a harmless no-op
        # (0 & 0 == 0) on a platform with none, but it still RUNS there, so this row is never
        # skipped and still guards the exact call site against a future regression.
        keypath.write_bytes(key_ctrlz)
        captured = {}
        real_open = R.os.open
        target = os.path.abspath(str(keypath))

        def capturing_open(path_, flags, *a, **kw):
            if os.path.abspath(os.fspath(path_)) == target:
                captured["flags"] = flags
            return real_open(path_, flags, *a, **kw)

        R.os.open = capturing_open
        try:
            R._probe_key(str(keypath))
        finally:
            R.os.open = real_open
        want_bit = getattr(os, "O_BINARY", 0)
        flags_ok = "flags" in captured and (captured["flags"] & want_bit) == want_bit
        r.ok("the probe's os.open flags include O_BINARY on a platform that defines one") \
            if flags_ok else r.fail("RESTORE-462 probe_key flags",
                                     f"captured={captured} want_bit={want_bit}")

        shutil.rmtree(home, ignore_errors=True)
    except Exception as e:
        r.fail("RESTORE-462 probe_key binary mode",
               f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
