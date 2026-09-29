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
the mistake in its own logs. Each row resets the module-level latch
(`_warned_put_kind_none_for_mask_shape`) itself, before and after, because a leaked "already
warned" flag from an earlier test would make it pass for the wrong reason.
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
    A kind=None mask at PREVIEW is stored exactly as it always was (choose_storage's own
    documented "stays eligible" default): the put that FIRES the warning, the put after the
    latch is set, and the same tensor as kind="IMAGE" all charge the same bytes and return
    the same values."""
    from TEX_Wrangle import tex_results
    from TEX_Wrangle import tex_packing

    tex_results._warned_put_kind_none_for_mask_shape = False
    try:
        mask = torch.rand(1, 8, 8, 1)
        c_fired, c_latched, c_image = (tex_results.ResultCache() for _ in range(3))
        fired = _capture(lambda: c_fired.put("m", mask, quality=tex_packing.PREVIEW, kind=None))
        latched = _capture(lambda: c_latched.put("m", mask, quality=tex_packing.PREVIEW,
                                                 kind=None))
        c_image.put("m", mask, quality=tex_packing.PREVIEW, kind="IMAGE")
        if not any("kind=None" in m for m in fired) or latched:
            r.fail("HOSTAUDIT-4b premise", f"warning did not fire once then latch: "
                   f"fired={fired} latched={latched}")
            return
        sizes = {n: c.stats()["ram_bytes"] for n, c in
                 (("fired", c_fired), ("latched", c_latched), ("image", c_image))}
        fp32_bytes = mask.numel() * mask.element_size()
        if len(set(sizes.values())) != 1 or sizes["fired"] >= fp32_bytes:
            r.fail("HOSTAUDIT-4b storage decision changed",
                   f"stored bytes {sizes} (fp32 would be {fp32_bytes}); the warning must not "
                   f"change the eligible-for-packing default")
        elif not (torch.equal(c_fired.get("m"), c_latched.get("m"))
                  and torch.equal(c_fired.get("m"), c_image.get("m"))):
            r.fail("HOSTAUDIT-4b stored values changed", "get() differs across the three puts")
        else:
            r.ok(f"kind=None mask at PREVIEW stores the same {sizes['fired']} bytes and values "
                 f"with the warning fired, latched, or as kind='IMAGE'")
    finally:
        tex_results._warned_put_kind_none_for_mask_shape = False
