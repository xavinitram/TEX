"""PERF-44 #1 — `tex_roi._walk`'s per-call SHA-256 is memoized.

WHAT WAS WRONG (MEASURE-44 finding #1). `_walk` computed `hashlib.sha256(code.encode())`
over the FULL source on every call, before it even looks its own memo key up — so a memo
HIT (the common case: the interactive path re-walks the same source 11 times a tick) still
paid a full re-hash of the source. `tex_cache.TEXCache.fingerprint` had already solved this
exact shape for its own (different) hash with `_FINGERPRINT_MEMO`; this is that memo's
sibling — `tex_cache.code_digest`, sharing its bounded-dict-then-clear discipline — reused
by `_walk` rather than a second one invented in `tex_roi.py`.

THE PROOF. Monkeypatch `hashlib.sha256` where `tex_cache.code_digest` actually calls it (not
where `_walk` used to call it directly — that call site is gone) and count real invocations
across many `_walk` calls on the SAME source: real hashing must happen exactly once. A
SECOND, different source must still cost its own hash — the memo is a cache, not a swallow.

PORTABILITY: pure Python, no ComfyUI, no CUDA, no compiler, no numpy, no timing assertion.
"""
import hashlib

from helpers import *

from TEX_Wrangle import tex_cache, tex_roi


def _reset():
    tex_roi.clear_roi_memo()          # _walk_memo + _region_dep_memo + _parse_memo
    tex_cache._CODE_DIGEST_MEMO.clear()


def test_perf44_walk_reuses_the_digest_on_a_memo_hit(r: SubTestResult):
    """RED-FIRST. N `_walk` calls on one unchanged source cost ONE real SHA-256, not N —
    even though every call after the first is itself a `_walk_memo` HIT (the shape the
    finding names: 'runs on EVERY call, memo hits included')."""
    print("\n--- PERF-44 #1: _walk's code digest is computed once per source ---")
    _reset()
    code = "@OUT = gauss_blur(@A, 2.0) + vec4($k);"
    real_sha256 = hashlib.sha256
    calls = {"n": 0}

    def counting_sha256(*a, **k):
        calls["n"] += 1
        return real_sha256(*a, **k)

    tex_cache.hashlib.sha256 = counting_sha256
    try:
        for i in range(20):
            tex_roi._walk(code, {"k": 1.0})   # identical call every time -> _walk_memo hit
    finally:
        tex_cache.hashlib.sha256 = real_sha256
    if calls["n"] != 1:
        r.fail("PERF-44 digest memo",
               f"20 identical _walk() calls triggered {calls['n']} real SHA-256(s), "
               f"expected 1 — the per-call digest is not memoized")
    else:
        r.ok("20 identical _walk() calls cost exactly 1 real SHA-256")

    # A second, DIFFERENT source is still hashed — the memo answers per-source, not globally.
    calls["n"] = 0
    other = "@OUT = gauss_blur(@A, 4.0) + vec4($k);"
    tex_cache.hashlib.sha256 = counting_sha256
    try:
        for i in range(5):
            tex_roi._walk(other, {"k": 1.0})
    finally:
        tex_cache.hashlib.sha256 = real_sha256
    if calls["n"] != 1:
        r.fail("PERF-44 digest memo (second source)",
               f"5 calls on a NEW source triggered {calls['n']} real SHA-256(s), expected 1")
    else:
        r.ok("a second, different source still costs exactly 1 real SHA-256 (memoized by "
             "source, not skipped entirely)")
    _reset()


def test_perf44_code_digest_matches_plain_sha256(r: SubTestResult):
    """`tex_cache.code_digest` answers exactly `hashlib.sha256(code.encode()).hexdigest()` —
    memoized, not a different value — for an arbitrary corpus of sources, cache cleared
    between each so a stale entry can't paper over a wrong hash."""
    print("\n--- PERF-44 #1: code_digest agrees with a fresh sha256 ---")
    bad = []
    for code in ("", "@OUT = @A;", "x" * 5000, "@OUT = gauss_blur(@A, $sigma);\n// comment\n"):
        tex_cache._CODE_DIGEST_MEMO.clear()
        got = tex_cache.code_digest(code)
        want = hashlib.sha256(code.encode()).hexdigest()
        if got != want:
            bad.append((code[:20], got, want))
    if bad:
        r.fail("PERF-44 code_digest value", f"{len(bad)} mismatch(es): {bad[:2]}")
    else:
        r.ok("code_digest(code) == sha256(code).hexdigest() across the corpus")
    tex_cache._CODE_DIGEST_MEMO.clear()


def test_perf44_code_digest_memo_is_bounded(r: SubTestResult):
    """The digest memo is a bounded cache (same discipline as `_FINGERPRINT_MEMO`): it clears
    itself rather than growing without limit."""
    print("\n--- PERF-44 #1: the digest memo is bounded ---")
    tex_cache._CODE_DIGEST_MEMO.clear()
    for i in range(tex_cache._CODE_DIGEST_MEMO_MAX + 20):
        tex_cache.code_digest(f"@OUT = vec4({i}.0);")
    size = len(tex_cache._CODE_DIGEST_MEMO)
    r.ok(f"digest memo bounded at {tex_cache._CODE_DIGEST_MEMO_MAX} (held {size})") \
        if size <= tex_cache._CODE_DIGEST_MEMO_MAX else \
        r.fail("PERF-44 digest memo bound",
               f"{size} entries exceeds cap {tex_cache._CODE_DIGEST_MEMO_MAX}")
    tex_cache._CODE_DIGEST_MEMO.clear()
