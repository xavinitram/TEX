"""TRK-144 — `tex_tool.py` redefined `tex_api._ver_tuple` as its own second copy (a third,
`tex_roi.py`, already imported `tex_api`'s). Both implementations agreed on every input
either module's real call sites ever passes it — `tex_api`'s two use sites always compare
a 2-component `X.Y` pragma/`LANGUAGE_VERSION` pair, `tex_tool`'s always compare either that
same 2-component pair (`tex_language`) or a 3-component `X.Y.Z` package-version pair
(`min_engine`) — but the two ALGORITHMS were not behaviourally identical in general:
`tex_api`'s truncated every input to exactly two components and collapsed the WHOLE tuple
to `(0, 0)` on any parse exception, while `tex_tool`'s kept one component per `.`-separated
chunk and degraded a bad chunk to `0` in place. Folding `tex_tool`'s copy away naively (by
having it import `tex_api`'s OLD 2-tuple algorithm) would have silently dropped the patch
component from `min_engine`/the package version — the TOOL-4 engine-version gate is the one
call site here that is not wrapped in a blanket `except Exception: pass`, so that would have
been a real, if obscure, regression in a load-bearing refusal, not a cosmetic one.

The fix instead moved `tex_tool`'s more general (per-chunk-tolerant, any-arity) algorithm
INTO `tex_api._ver_tuple` as the one definition, and had `tex_tool.load_tool` import it
(unconditionally, ahead of the try/except that also imports `LANGUAGE_VERSION`, so a
`tex_api` import failure there cannot leave the later `min_engine` gate calling an unbound
name). This is behaviour-identical for every real call site because `tex_api`'s own two
sites only ever see well-formed 2-component strings (regex-anchored by
`tex_compiler.parser.language_pragma`, or the `LANGUAGE_VERSION` constant), for which the
general algorithm produces the exact same 2-tuple the old one did, and `tex_tool`'s own two
sites keep running literally the same per-chunk code as before, just homed elsewhere.

ComfyUI-invisible because: no call path, default or behaviour moves for any well-formed
version string a real manifest, pragma or package version ever carries — this only changes
which module OWNS one already-shared algorithm, and closes a latent divergence that no
shipped input could reach today (`min_engine` is not schema-validated to be numeric, so the
gap was real, just unexercised).
"""
from helpers import *

from TEX_Wrangle import tex_api, tex_tool


def test_trk144_tex_tool_no_longer_redefines_ver_tuple(r: SubTestResult):
    """`tex_tool` carries no module-level `_ver_tuple` of its own any more — the only
    definition left in the tree is `tex_api._ver_tuple`, imported where needed."""
    print("\n--- TRK-144: tex_tool has no _ver_tuple of its own; tex_api's is the one home ---")
    try:
        assert not hasattr(tex_tool, "_ver_tuple"), (
            "tex_tool still carries its own module-level _ver_tuple — the redefinition "
            "TRK-144 asked to remove is still there")
        assert hasattr(tex_api, "_ver_tuple"), "tex_api lost its _ver_tuple entirely"
        r.ok("tex_tool has no module-level _ver_tuple; tex_api._ver_tuple is the one definition")
    except AssertionError as e:
        r.fail("TRK-144 single definition", str(e))


def test_trk144_ver_tuple_examples(r: SubTestResult):
    """The unified algorithm: one int per `.`-separated component, a non-numeric-leading
    chunk degrades to 0 IN PLACE (never collapses the whole tuple), matching the
    behaviour `tex_tool`'s min_engine/tex_language gates always relied on."""
    print("\n--- TRK-144: tex_api._ver_tuple parses per-chunk, any arity ---")
    cases = [
        ("0.25", (0, 25)),
        ("0.40.2", (0, 40, 2)),          # 3-component: the patch digit must survive
        ("1", (1,)),
        ("1.abc", (1, 0)),               # a bad chunk degrades in place, not to a sentinel
        ("", (0,)),
    ]
    for v, want in cases:
        try:
            got = tex_api._ver_tuple(v)
            assert got == want, f"_ver_tuple({v!r}) = {got}, expected {want}"
            r.ok(f"_ver_tuple({v!r}) == {want}")
        except AssertionError as e:
            r.fail(f"_ver_tuple({v!r})", str(e))


def test_trk144_min_engine_gate_is_patch_precise(r: SubTestResult):
    """The load-bearing (non-advisory) TOOL-4 engine-version gate in `load_tool` must still
    tell a newer PATCH apart, not just a newer major/minor — this is exactly the precision
    that would have been silently lost had `tex_tool` naively imported `tex_api`'s OLD
    2-tuple-truncating algorithm instead of the other way around."""
    print("\n--- TRK-144: min_engine gate stays patch-precise after the dedup ---")
    from TEX_Wrangle import __version__ as pkg_version
    base = {"manifest_schema": 1, "name": "X", "tex_language": "0.23",
            "code": "@OUT = @image;", "inputs": [{"name": "image", "type": "IMAGE"}],
            "promoted_params": []}
    parts = pkg_version.split(".")
    bumped_patch = ".".join(parts[:2] + [str(int(parts[2]) + 1)]) if len(parts) > 2 else "999.0.0"

    try:
        # min_engine one patch AHEAD of the running package -> must still refuse.
        try:
            tex_tool.load_tool({**base, "min_engine": bumped_patch})
            r.fail("TRK-144 patch-precise gate",
                   f"load_tool accepted min_engine={bumped_patch!r} against package "
                   f"{pkg_version!r} — patch-level precision was lost")
        except tex_tool.TEXToolError:
            r.ok(f"min_engine={bumped_patch!r} > package {pkg_version!r} still refused")

        # min_engine equal to the running package -> must still load.
        m = tex_tool.load_tool({**base, "min_engine": pkg_version})
        assert m.min_engine == pkg_version
        r.ok(f"min_engine={pkg_version!r} == package version still loads")
    except Exception as e:
        r.fail("TRK-144 min_engine gate", f"{type(e).__name__}: {e}")


def test_trk144_language_pin_advisory_unchanged(r: SubTestResult):
    """The LANG-3 advisory (tex_language ahead of LANGUAGE_VERSION) still fires, and still
    does not block loading — unchanged by where `_ver_tuple` now lives."""
    print("\n--- TRK-144: LANG-3 tex_language advisory unchanged ---")
    base = {"manifest_schema": 1, "name": "X",
            "code": "@OUT = @image;", "inputs": [{"name": "image", "type": "IMAGE"}],
            "promoted_params": []}
    try:
        m = tex_tool.load_tool({**base, "tex_language": "999.0"})
        assert m.tex_language == "999.0", m.tex_language
        r.ok("a tex_language far ahead of LANGUAGE_VERSION does not block load_tool")
    except Exception as e:
        r.fail("TRK-144 language advisory", f"{type(e).__name__}: {e}")
