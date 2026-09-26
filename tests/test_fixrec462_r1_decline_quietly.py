"""FIX-REC (v0.46.2 Phase C) — R1 [altitude]: verdict->action in ONE place.

Before this, `tex_cache._load_from_disk`, `tex_cache._load_codegen_from_disk` and
`tex_results._restore` each hand-spelled `is _FUTURE_TRAILER or is _UNREADABLE` to mean the
same thing ("a miss, but never delete the file") — RESTORE-462's own diff already landed the
identical line twice (R2-simplification finding #5; R4-altitude finding #1). The fix is one
predicate, `tex_recovery._is_decline_quietly`, that the three sites now call instead of
re-deriving the grouping.

This is a pure refactor (no call site's return shape changed), so there is no red-first mutation
here — the proof is that all three sites still classify FUTURE_TRAILER and UNREADABLE exactly as
before (miss, file survives), and that `_UNVERIFIED` is unaffected (still a miss that MAY delete).
"""
from helpers import *

import builtins

from TEX_Wrangle.tex_cache import TEXCache
from TEX_Wrangle import tex_results
from TEX_Wrangle import tex_recovery as R


def _frame(res=32):
    return torch.rand(1, res, res, 4)


def _future_trailer_bytes() -> bytes:
    """Any payload followed by a `TEXm<n>` trailer where `n != 1` — `load_verified` classes
    this FUTURE_TRAILER before it ever computes a MAC, so the payload content is irrelevant."""
    return b"whatever-a-newer-tex-wrote" + b"TEXm9" + b"\x00" * 32


def test_r1_is_decline_quietly_predicate(r: SubTestResult):
    print("\n--- FIX-REC R1: _is_decline_quietly classifies the four verdict-shaped values ---")
    checks = [
        (R._FUTURE_TRAILER, True, "_FUTURE_TRAILER"),
        (R._UNREADABLE, True, "_UNREADABLE"),
        (R._UNVERIFIED, False, "_UNVERIFIED (may delete — not part of this group)"),
        ({"t": _frame()}, False, "an ordinary deserialised record"),
        (None, False, "None"),
    ]
    ok = True
    for value, want, label in checks:
        got = R._is_decline_quietly(value)
        if got != want:
            ok = False
            r.fail("FIX-REC R1 predicate", f"{label}: expected {want}, got {got}")
    if ok:
        r.ok("_is_decline_quietly(FUTURE_TRAILER/UNREADABLE)=True, "
             "_is_decline_quietly(UNVERIFIED/other)=False")


def test_r1_predicate_never_raises_on_a_tensor(r: SubTestResult):
    """The identity-only implementation must never fall into `==` against a tensor (which would
    return an elementwise tensor, ambiguous in a boolean context) — the exact trap a naive
    `verdict in (_FUTURE_TRAILER, _UNREADABLE)` membership test would risk for a caller that ever
    hands it a raw tensor instead of a record dict."""
    print("\n--- FIX-REC R1: the predicate is identity-only, never falls into tensor == ---")
    try:
        t = _frame()
        got = R._is_decline_quietly(t)
        if got is False:
            r.ok("a tensor input classifies False without raising")
        else:
            r.fail("FIX-REC R1 predicate tensor", f"expected False, got {got!r}")
    except Exception as e:
        r.fail("FIX-REC R1 predicate tensor", f"{type(e).__name__}: {e}")


def test_r1_pkl_site_declines_future_trailer_quietly(r: SubTestResult):
    print("\n--- FIX-REC R1: _load_from_disk leaves a FUTURE_TRAILER .pkl on disk, as a miss ---")
    try:
        d = Path(tempfile.mkdtemp())
        cache = TEXCache(cache_dir=d)
        bt = {"A": TEXType.VEC4}
        fp = cache.fingerprint("@OUT = @A * 0.7 + 0.1;", bt)
        path = cache._disk_path(fp)
        path.write_bytes(_future_trailer_bytes())
        got = cache._load_from_disk(fp, bt)
        ok = got is None and path.exists()
        r.ok("a FUTURE_TRAILER .pkl is a miss and survives on disk") if ok else \
            r.fail("FIX-REC R1 .pkl future-trailer",
                   f"load_returned_non_none={got is not None} survived={path.exists()}")
        shutil.rmtree(d, ignore_errors=True)
    except Exception as e:
        r.fail("FIX-REC R1 .pkl future-trailer", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


def test_r1_cg_site_declines_future_trailer_quietly(r: SubTestResult):
    print("\n--- FIX-REC R1: _load_codegen_from_disk leaves a FUTURE_TRAILER .cg on disk ---")
    try:
        d = Path(tempfile.mkdtemp())
        cache = TEXCache(cache_dir=d)
        fp = "cafef00d" * 8
        path = cache._cg_path(fp)
        path.write_bytes(_future_trailer_bytes())
        got = cache._load_codegen_from_disk(fp)
        ok = got is None and path.exists()
        r.ok("a FUTURE_TRAILER .cg is a miss and survives on disk") if ok else \
            r.fail("FIX-REC R1 .cg future-trailer",
                   f"load_returned_non_none={got is not None} survived={path.exists()}")
        shutil.rmtree(d, ignore_errors=True)
    except Exception as e:
        r.fail("FIX-REC R1 .cg future-trailer", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


def test_r1_frame_site_declines_future_trailer_quietly(r: SubTestResult):
    print("\n--- FIX-REC R1: _restore leaves a FUTURE_TRAILER .frame on disk, as a miss ---")
    try:
        d = Path(tempfile.mkdtemp())
        c = tex_results.ResultCache(cache_dir=str(d), budget_mb=0)
        path = Path(c._disk_path("k"))
        path.write_bytes(_future_trailer_bytes())
        with c._lock:
            c._spilled = None
        frame, _orig, _fence = c._restore("k")
        ok = frame is None and path.exists()
        r.ok("a FUTURE_TRAILER .frame is a miss and survives on disk") if ok else \
            r.fail("FIX-REC R1 .frame future-trailer",
                   f"restore_returned_non_none={frame is not None} survived={path.exists()}")
        shutil.rmtree(d, ignore_errors=True)
    except Exception as e:
        r.fail("FIX-REC R1 .frame future-trailer", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


def test_r1_pkl_and_cg_sites_still_decline_unreadable_quietly(r: SubTestResult):
    """The `_restore` site's UNREADABLE behaviour is already pinned by RESTORE-462's own
    `test_restore462_transient_open_failure_is_a_miss_not_a_deletion`; this covers the other two
    sites the refactor also touches, so all three keep the same non-delete-on-transient-failure
    behaviour after routing through the shared predicate."""
    print("\n--- FIX-REC R1: the .pkl/.cg sites also leave an UNREADABLE file on disk ---")
    real_open = builtins.open
    try:
        d = Path(tempfile.mkdtemp())
        cache = TEXCache(cache_dir=d)
        bt = {"A": TEXType.VEC4}
        fp = cache.fingerprint("@OUT = @A * 0.7 + 0.1;", bt)
        pkl_path = cache._disk_path(fp)
        pkl_path.write_bytes(b"irrelevant, the open() itself will fail")
        cg_fp = "0ddba11" * 8 + "0"
        cg_path = cache._cg_path(cg_fp)
        cg_path.write_bytes(b"irrelevant, the open() itself will fail")

        def _flaky(target):
            state = {"raised": False}
            tgt = os.path.abspath(str(target))

            def flaky_open(file, *a, **kw):
                if not state["raised"] and os.path.abspath(os.fspath(file)) == tgt:
                    state["raised"] = True
                    raise PermissionError(13, "in use by another process")
                return real_open(file, *a, **kw)
            return flaky_open

        builtins.open = _flaky(pkl_path)
        try:
            got_pkl = cache._load_from_disk(fp, bt)
        finally:
            builtins.open = real_open
        pkl_ok = got_pkl is None and pkl_path.exists()

        builtins.open = _flaky(cg_path)
        try:
            got_cg = cache._load_codegen_from_disk(cg_fp)
        finally:
            builtins.open = real_open
        cg_ok = got_cg is None and cg_path.exists()

        if pkl_ok and cg_ok:
            r.ok("both the .pkl and .cg sites leave a transiently-unreadable file on disk")
        else:
            r.fail("FIX-REC R1 unreadable",
                   f"pkl: hit={got_pkl is not None} survived={pkl_path.exists()}; "
                   f"cg: hit={got_cg is not None} survived={cg_path.exists()}")
        shutil.rmtree(d, ignore_errors=True)
    except Exception as e:
        builtins.open = real_open
        r.fail("FIX-REC R1 unreadable", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
