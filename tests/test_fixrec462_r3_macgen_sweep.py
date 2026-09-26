"""FIX-REC (v0.46.2 Phase C) — R3 [LOW]: reclaim `.macgen-*.tmp` orphans from a crash mid-mint
(B3#1).

`_publish_new_key` mints the per-user MAC key into a `.macgen-*.tmp` temp (via
`bounded_mkstemp`) and only THEN publishes it atomically. If the process dies (crash, SIGKILL,
power loss) after the temp is written but before that publish step, the temp is orphaned —
and nothing anywhere reclaimed it: `sweep_temps` (this module's other reclaim) matches a
different prefix and only ever runs against the CACHE-2 spill directory, never the key home.
The fix sweeps stale `.macgen-*.tmp` files in the key home on the way into
`_resolve_or_create_key` (every `_mac_key()` call, i.e. on both minting and probing), age-gated
so a peer's in-flight mint — microseconds old — is never mistaken for an orphan.
"""
from helpers import *

import time

from TEX_Wrangle import tex_recovery as R


def test_r3_crash_orphaned_macgen_temp_is_reclaimed(r: SubTestResult):
    print("\n--- FIX-REC R3: a stale .macgen-*.tmp is swept on the next key resolve ---")
    try:
        home = Path(tempfile.mkdtemp())
        # The exact shape a crash mid-mint leaves: `_publish_new_key`'s own prefix/suffix,
        # aged past the grace window (a real orphan, not a peer mid-mint).
        orphan = home / (R._MACGEN_PREFIX + "deadbeef.tmp")
        orphan.write_bytes(b"\x00" * R._MAC_KEY_LEN)
        old = time.time() - R._MACGEN_ORPHAN_GRACE_SEC - 5
        os.utime(orphan, (old, old))

        orig_home = R._mac_key_home
        R._mac_key_home = lambda: str(home)
        try:
            key = R._resolve_or_create_key()
        finally:
            R._mac_key_home = orig_home

        ok = key is not None and not orphan.exists()
        r.ok("a crash-orphaned .macgen-*.tmp is reclaimed on the next key resolve") if ok \
            else r.fail("FIX-REC R3 orphan sweep",
                        f"key_resolved={key is not None} orphan_survived={orphan.exists()}")
        shutil.rmtree(home, ignore_errors=True)
    except Exception as e:
        r.fail("FIX-REC R3 orphan sweep", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


def test_r3_a_peers_fresh_macgen_temp_is_never_removed(r: SubTestResult):
    """The age-gate's other direction: a `.macgen-*.tmp` younger than the grace window looks
    exactly like a peer's in-flight mint and must survive the sweep — an eager reclaim here
    would let two processes converge on different keys if one's publish is merely slow (a
    loaded box, not a crash)."""
    print("\n--- FIX-REC R3: a fresh .macgen-*.tmp (a peer's in-flight mint) survives ---")
    try:
        home = Path(tempfile.mkdtemp())
        fresh = home / (R._MACGEN_PREFIX + "feedface.tmp")
        fresh.write_bytes(b"\x00" * R._MAC_KEY_LEN)   # mtime = now, well inside the grace window

        orig_home = R._mac_key_home
        R._mac_key_home = lambda: str(home)
        try:
            R._resolve_or_create_key()
        finally:
            R._mac_key_home = orig_home

        ok = fresh.exists()
        r.ok("a fresh (peer in-flight) .macgen-*.tmp is left alone by the sweep") if ok else \
            r.fail("FIX-REC R3 age-gate", "a fresh .macgen-*.tmp was removed")
        shutil.rmtree(home, ignore_errors=True)
    except Exception as e:
        r.fail("FIX-REC R3 age-gate", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


def test_r3_sweep_helper_reports_count_and_ignores_other_names(r: SubTestResult):
    """Unit-level check of `_sweep_stale_macgen_temps` itself: it only ever touches its own
    prefix+suffix (never `sweep_temps`'s `.tex-tmp-*.tmp`, never an unrelated file), counts what
    it removed, and tolerates a directory that does not exist."""
    print("\n--- FIX-REC R3: _sweep_stale_macgen_temps is name-scoped and reports its count ---")
    try:
        home = Path(tempfile.mkdtemp())
        old = time.time() - R._MACGEN_ORPHAN_GRACE_SEC - 5

        stale_macgen = home / (R._MACGEN_PREFIX + "aaaaaaaa.tmp")
        stale_macgen.write_bytes(b"x")
        os.utime(stale_macgen, (old, old))

        stale_other_prefix = home / (R.TMP_PREFIX + "aaaaaaaa.tmp")   # a different reclaim's name
        stale_other_prefix.write_bytes(b"x")
        os.utime(stale_other_prefix, (old, old))

        unrelated = home / "cache_mac.key"
        unrelated.write_bytes(b"y" * R._MAC_KEY_LEN)
        os.utime(unrelated, (old, old))

        n = R._sweep_stale_macgen_temps(str(home))

        ok = (n == 1 and not stale_macgen.exists()
              and stale_other_prefix.exists() and unrelated.exists())
        if ok:
            r.ok("the sweep removes exactly its own stale temp and nothing else")
        else:
            r.fail("FIX-REC R3 sweep scope",
                   f"n={n} macgen_gone={not stale_macgen.exists()} "
                   f"other_prefix_survived={stale_other_prefix.exists()} "
                   f"key_survived={unrelated.exists()}")

        n2 = R._sweep_stale_macgen_temps(str(home / "does-not-exist"))
        if n2 != 0:
            r.fail("FIX-REC R3 sweep missing dir", f"expected 0, got {n2}")
        else:
            r.ok("sweeping a nonexistent directory is a no-op, not an error")
        shutil.rmtree(home, ignore_errors=True)
    except Exception as e:
        r.fail("FIX-REC R3 sweep scope", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
