"""SCALE-47b phase 6b — tex_api.scale_verdict(): a cheap, memoized, pre-cook query.

Orchestrator addition (accepted 2026-09-26): a host wants to ask "would a non-trivial scale
be refused for this program?" BEFORE cooking, at negligible cost on every drag tick, and the
answer must be the exact same verdict `tex_engine.prepare()`'s refusal path uses -- the two
must never be able to disagree, because that would make the pre-cook query a lie. Both read
`tex_roi.scale_verdict()`'s one memoized answer per (source, $param values).
"""
from helpers import *
from TEX_Wrangle import tex_api
from TEX_Wrangle import tex_roi as R
from TEX_Wrangle import tex_engine

_SAFE_CODE = "@OUT = gauss_blur(@A, 4.0);"
_UNSAFE_CODE = "@OUT = vec4(ix * 0.01, iy * 0.01, 0.0, 1.0);"
_PRAGMA_SAFE_CODE = "//!tex scale: safe\n" + _UNSAFE_CODE
_PRAGMA_NEVER_CODE = "//!tex scale: never\n" + _SAFE_CODE


def test_scale47b_verdict_shape(r: SubTestResult):
    print("\n--- SCALE-47b: tex_api.scale_verdict() returns (safe, code, source) ---")
    v = tex_api.scale_verdict(_SAFE_CODE)
    if not hasattr(v, "safe") or not hasattr(v, "code") or not hasattr(v, "source"):
        r.fail("verdict shape", f"expected .safe/.code/.source, got {v!r}")
        return
    r.ok(f"tex_api.scale_verdict({_SAFE_CODE!r}) -> {v!r}")


def test_scale47b_verdict_four_cases(r: SubTestResult):
    print("\n--- SCALE-47b: the four verdict shapes (classifier x2, pragma x2) ---")
    cases = [
        (_SAFE_CODE, True, None, "classifier"),
        (_UNSAFE_CODE, False, "scale-unsafe", "classifier"),
        (_PRAGMA_SAFE_CODE, True, None, "pragma_safe"),
        (_PRAGMA_NEVER_CODE, False, "scale-unsafe", "pragma_never"),
    ]
    for code, safe, code_, source in cases:
        v = tex_api.scale_verdict(code)
        if (v.safe, v.code, v.source) != (safe, code_, source):
            r.fail("verdict case", f"for {code!r}: expected "
                   f"(safe={safe}, code={code_!r}, source={source!r}), got {v!r}")
            return
    r.ok("all four verdict shapes (classifier-safe, classifier-unsafe, pragma_safe, "
         "pragma_never) match")


def test_scale47b_verdict_agrees_with_cook_refusal(r: SubTestResult):
    print("\n--- SCALE-47b: scale_verdict() and prepare()'s refusal can never disagree ---")
    A = make_img(1, 8, 8, 4)
    for code in (_SAFE_CODE, _UNSAFE_CODE, _PRAGMA_SAFE_CODE, _PRAGMA_NEVER_CODE):
        v = tex_api.scale_verdict(code)
        raised = None
        try:
            tex_engine.prepare(code, {"A": A.clone()}, device_mode="cpu", scale=0.5)
        except Exception as e:
            raised = e
        cook_says_safe = raised is None
        if cook_says_safe != v.safe:
            r.fail("verdict/cook agreement",
                   f"for {code!r}: scale_verdict says safe={v.safe}, but prepare() "
                   f"{'did not raise' if cook_says_safe else 'raised: ' + str(raised)}")
            return
        if not v.safe:
            refusal = getattr(raised, "tex_refusal", None)
            if refusal is None or refusal.code != v.code:
                r.fail("verdict/cook code agreement",
                       f"for {code!r}: scale_verdict.code={v.code!r} but "
                       f"tex_refusal.code={getattr(refusal, 'code', None)!r}")
                return
    r.ok("scale_verdict() and prepare()'s refusal agree on all four cases, code included")


def test_scale47b_verdict_memoized_no_reanalysis(r: SubTestResult):
    print("\n--- SCALE-47b: a second scale_verdict() call on the same program re-analyzes nothing ---")
    code = "@OUT = gauss_blur(@A, vec4(9.0).r);"  # a program unlikely to collide with other tests
    calls = {"n": 0}
    real_walk = R._scale_unsafe_walk

    def _counting_walk(node, in_coord_arg=False):
        calls["n"] += 1
        return real_walk(node, in_coord_arg=in_coord_arg)

    R._scale_unsafe_walk = _counting_walk
    try:
        R._scale_verdict_memo.clear()
        first = tex_api.scale_verdict(code)
        n_after_first = calls["n"]
        if n_after_first == 0:
            r.fail("counting setup", "the walk was never called on the first (cold) lookup")
            return
        second = tex_api.scale_verdict(code)
        if calls["n"] != n_after_first:
            r.fail("verdict memoization",
                   f"a second scale_verdict() call on the same program re-analyzed it: "
                   f"walk calls went from {n_after_first} to {calls['n']}")
            return
        if second != first:
            r.fail("verdict identity", f"{first!r} != {second!r}")
            return
        r.ok(f"cold lookup called the walk {n_after_first} time(s); a warm repeat called it "
             f"0 more times")
    finally:
        R._scale_unsafe_walk = real_walk
        R._scale_verdict_memo.clear()
