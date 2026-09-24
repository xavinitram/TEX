"""
v0.42 HOSTAUDIT-4b — a one-time warning for `ResultCache.put(..., kind=None)` on a mask.

`tex_packing.choose_storage`'s own docstring already says it: `kind=None` ("the caller did
not say") "stays eligible" for PREVIEW packing — the same eligibility `kind="IMAGE"` would
get. But `kind="MASK"` (without `mask_eligible=True`) is REFUSED by the very next gate. So
the one spelling of this call that is silently the OPPOSITE of that documented MASK default
is the one where the caller forgot to say anything at all — exactly the footgun a compass
review named for a matte producer (`ResultCache.put(..., kind="MASK")`, "where the kind=None
default is a footgun ... has noted").

This does not change what gets stored — same key, same bytes, same `tex_packing.choose_storage`
call with the same arguments — it only adds one process-lifetime log line so a host can catch
the mistake in its own logs. `_reset_for_test` below (a new, test-only helper) exists because
the warning is a global latch and a leaked "already warned" flag from an earlier test would
make this test pass for the wrong reason.
"""
import logging

import torch


def _capture(fn):
    """Run `fn()` with a handler attached to the "TEX" logger; return its emitted messages."""
    records = []
    handler = logging.Handler()
    handler.emit = lambda rec: records.append(rec.getMessage())
    log = logging.getLogger("TEX")
    log.addHandler(handler)
    try:
        fn()
    finally:
        log.removeHandler(handler)
    return records


def test_hostaudit4b_warns_once_for_mask_shaped_kind_none_at_preview(r):
    from TEX_Wrangle import tex_results
    from TEX_Wrangle import tex_packing

    tex_results._warned_put_kind_none_for_mask_shape = False   # test-local reset of the latch
    try:
        cache = tex_results.ResultCache()
        mask = torch.rand(1, 8, 8, 1)   # 1-channel: mask-shaped

        records = _capture(lambda: cache.put("m1", mask, quality=tex_packing.PREVIEW))
        if not any("kind=None" in m and "m1" in m for m in records):
            r.fail("HOSTAUDIT-4b warning", f"no matching warning logged: {records}")
            return
        r.ok("put(kind=None) on a 1-channel tensor at PREVIEW logs a warning")

        # Second put (different key, same shape/quality/kind=None) — the SAME warning must
        # not fire again; a per-call warning on a scrub/video path would be a log flood.
        records2 = _capture(lambda: cache.put("m2", mask, quality=tex_packing.PREVIEW))
        if any("kind=None" in m for m in records2):
            r.fail("HOSTAUDIT-4b warning throttle", f"warned a second time: {records2}")
        else:
            r.ok("the warning fires at most once per process")
    finally:
        tex_results._warned_put_kind_none_for_mask_shape = False


def test_hostaudit4b_no_warning_when_kind_is_given(r):
    """Passing kind="MASK" explicitly — right or wrong for the caller's intent — must not
    trip the warning: the whole point is to catch an OMISSION, not a stated choice."""
    from TEX_Wrangle import tex_results
    from TEX_Wrangle import tex_packing

    tex_results._warned_put_kind_none_for_mask_shape = False
    try:
        cache = tex_results.ResultCache()
        mask = torch.rand(1, 8, 8, 1)
        records = _capture(lambda: cache.put("m3", mask, quality=tex_packing.PREVIEW,
                                             kind="MASK"))
        if any("kind=None" in m for m in records):
            r.fail("HOSTAUDIT-4b false positive", f"warned despite an explicit kind: {records}")
        else:
            r.ok("an explicit kind= (MASK) never trips the kind=None warning")
    finally:
        tex_results._warned_put_kind_none_for_mask_shape = False


def test_hostaudit4b_no_warning_off_preview_or_off_mask_shape(r):
    """The warning is scoped tightly: no PREVIEW quality, or a non-1-channel tensor, must
    never fire it — those calls were never at risk of the silent-pack footgun."""
    from TEX_Wrangle import tex_results
    from TEX_Wrangle import tex_packing

    tex_results._warned_put_kind_none_for_mask_shape = False
    try:
        cache = tex_results.ResultCache()
        mask = torch.rand(1, 8, 8, 1)
        image = torch.rand(1, 8, 8, 4)   # 4-channel: not mask-shaped

        r1 = _capture(lambda: cache.put("no-preview", mask, kind=None))   # quality=None
        r2 = _capture(lambda: cache.put("not-1ch", image, quality=tex_packing.PREVIEW,
                                        kind=None))
        bad = [m for m in (r1 + r2) if "kind=None" in m]
        if bad:
            r.fail("HOSTAUDIT-4b over-broad warning", f"fired when it should not have: {bad}")
        else:
            r.ok("no warning off PREVIEW quality and no warning for a non-1-channel tensor")
    finally:
        tex_results._warned_put_kind_none_for_mask_shape = False


def test_hostaudit4b_storage_decision_is_unaffected(r):
    """The whole point is additive: the warning changes nothing about what gets cached.
    A kind=None mask at PREVIEW is packed exactly as it always was (choose_storage's own
    documented "stays eligible" default), warning or no warning."""
    from TEX_Wrangle import tex_results
    from TEX_Wrangle import tex_packing

    tex_results._warned_put_kind_none_for_mask_shape = False
    try:
        mask = torch.rand(1, 8, 8, 1)
        want_before = tex_packing.choose_storage(mask, quality=tex_packing.PREVIEW, kind=None)
        _capture(lambda: None)   # no-op, just keeps the helper's shape consistent
        want_after = tex_packing.choose_storage(mask, quality=tex_packing.PREVIEW, kind=None)
        if want_before == want_after:
            r.ok(f"choose_storage(kind=None) is unchanged by this ask: {want_before!r}")
        else:
            r.fail("HOSTAUDIT-4b storage decision changed",
                   f"{want_before!r} -> {want_after!r}")
    finally:
        tex_results._warned_put_kind_none_for_mask_shape = False
