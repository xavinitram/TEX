"""HOOK-2 — a per-document binding map on tex_lsp's diagnostics path.

The complaint: `diagnostics_for` checked every document against a hardcoded `{}`, so every
`@input` resolved to VEC4 and a real narrower-typed input's out-of-range channel access
type-checked clean here and only failed at cook. A host that lint-checks in process has no
such gap — it can call `tex_api.check(source, binding_types)` directly, against whatever it
actually has wired — which is exactly the reason to give the wire path the same map: the
fix adds an optional `bindingTypes` field to `didOpen`/`didChange` params, the wire form of
the same `{name: TEXType}` map `check()` already takes, so the wire path and a direct
in-process call now type-check against one shape.

Four shapes, one test each: a VEC4-masks-a-real-error example, WITH and WITHOUT the map;
byte-for-byte parity with the pre-HOOK-2 hardcoded behaviour when no map is ever sent;
didChange persistence (a map survives a text-only edit, and a bindings-only edit still
republishes rather than going stale); and total-server safety over a battery of malformed
`bindingTypes` shapes, plus `_parse_binding_types`'s own parse contract directly.
"""
from helpers import *

from TEX_Wrangle import tex_api, tex_lsp
from TEX_Wrangle.tex_compiler.types import TEXType

# The motivating shape: `@image.a` on a real 3-channel input. Against {}, @image defaults to
# VEC4 and `.a` (channel index 3) is in range -- clean. Against a map that says @image is
# VEC3, `.a` is out of range -- E3301, exactly the diagnostic {} hides.
_PROG = "@OUT = vec4(@image.a, @image.a, @image.a, 1.0);"


def _error_codes(diags):
    return [d["code"] for d in diags if d["severity"] == 1]


# ── the want: a wired 3-channel input surfaces what {} hides ─────────────────

def test_hook2_binding_map_surfaces_the_vec4_assumption(r: SubTestResult):
    print("\n--- HOOK-2: a wired 3-channel input surfaces the diagnostic {} hides ---")

    # (1) tex_api.check itself, in-process -- the premise, and the same {name: TEXType}
    # shape a host's own in-process lint call would build.
    clean = tex_api.check(_PROG, {})
    narrowed = tex_api.check(_PROG, {"image": TEXType.VEC3})
    if not any(d.code == "E3301" for d in clean):
        r.ok("premise: check() against {} does not flag @image.a (VEC4 default)")
    else:
        r.fail("HOOK-2 premise", "the example no longer type-checks clean against {}")
    if any(d.code == "E3301" and d.severity == "error" for d in narrowed):
        r.ok("check() against a VEC3 map DOES flag @image.a (E3301)")
    else:
        r.fail("HOOK-2 check()", f"VEC3-mapped check() did not raise E3301: {narrowed}")

    # (2) the same two shapes, over the wire: didOpen with/without bindingTypes.
    s = tex_lsp.LSPServer()
    _, no_map = s.handle("textDocument/didOpen",
                          {"textDocument": {"uri": "a", "text": _PROG}})
    _, with_map = s.handle("textDocument/didOpen",
                            {"textDocument": {"uri": "b", "text": _PROG},
                             "bindingTypes": {"image": "vec3"}})
    diags_no_map = no_map[0]["params"]["diagnostics"]
    diags_with_map = with_map[0]["params"]["diagnostics"]
    if "E3301" not in _error_codes(diags_no_map):
        r.ok("wire, no bindingTypes: clean, same as {}")
    else:
        r.fail("HOOK-2 wire no-map", f"got {diags_no_map}")
    if "E3301" in _error_codes(diags_with_map):
        r.ok("wire, bindingTypes={'image': 'vec3'}: E3301 surfaces")
    else:
        r.fail("HOOK-2 wire with-map", f"got {diags_with_map}")


# ── the invisibility half: no map ever sent == the base sha, dict-for-dict ───

def _base_sha_diagnostics_for(source):
    """The pre-HOOK-2 body of `diagnostics_for`, kept here as a reference so a regression
    in the new default path is caught by comparison, not by re-reading the diff by eye."""
    out = []
    for d in tex_api.check(source, {}):
        dd = d.to_dict()
        line = max(0, (dd.get("line") or 1) - 1)
        col = max(0, (dd.get("col") or 1) - 1)
        end_line = max(0, (dd.get("end_line") or dd.get("line") or 1) - 1)
        end_col = dd.get("end_col")
        end_col = col + 1 if end_col is None else max(0, end_col - 1)
        item = {
            "range": {"start": {"line": line, "character": col},
                      "end": {"line": end_line, "character": end_col}},
            "severity": {"error": 1, "warning": 2, "info": 3, "hint": 4}.get(
                dd.get("severity", "error"), 1),
            "code": dd.get("code", ""),
            "source": "tex",
            "message": dd.get("message", ""),
        }
        if dd.get("docs_url"):
            item["codeDescription"] = {"href": dd["docs_url"]}
        out.append(item)
    return out


def test_hook2_no_map_is_byte_for_byte_the_base_sha(r: SubTestResult):
    print("\n--- HOOK-2: no map sent -> identical to the base-sha hardcoded check(src, {}) ---")
    progs = [
        _PROG,                                       # clean against {}
        "@OUT = nosuchfn(@image);",                   # E5001 -- unknown function
        "@OUT = vec4(@image.rgb, 1.0);",              # clean, VEC4 base
        "float x = 0.5; @OUT = vec4(x.rgb, 1.0);",    # E3301 on a scalar -- unrelated to HOOK-2
        "float unused = 1.0; @OUT = @image;",         # W7001 warning (severity, not just error)
    ]
    all_match = True
    for p in progs:
        got = tex_lsp.diagnostics_for(p)              # no binding_types argument at all
        want = _base_sha_diagnostics_for(p)
        if got != want:
            all_match = False
            r.fail("HOOK-2 base-sha parity", f"{p!r}: got {got}, wanted {want}")
    if all_match:
        r.ok(f"diagnostics_for(source) with no map matches the base-sha check(source, {{}}) "
             f"on {len(progs)} programs, dict-for-dict (errors and warnings)")

    # And through the full handle() dispatch, where the wire never mentions bindingTypes.
    s = tex_lsp.LSPServer()
    _, notes = s.handle("textDocument/didOpen", {"textDocument": {"uri": "u", "text": _PROG}})
    if notes[0]["params"]["diagnostics"] == _base_sha_diagnostics_for(_PROG):
        r.ok("handle('textDocument/didOpen', ...) with no bindingTypes key matches the base sha")
    else:
        r.fail("HOOK-2 handle() parity", f"got {notes[0]['params']['diagnostics']}")


# ── didChange: the map persists like the text does ───────────────────────────

def test_hook2_didchange_without_a_map_keeps_the_previous_map(r: SubTestResult):
    print("\n--- HOOK-2: didChange without bindingTypes keeps the document's last map ---")
    s = tex_lsp.LSPServer()
    s.handle("textDocument/didOpen",
             {"textDocument": {"uri": "u", "text": _PROG},
              "bindingTypes": {"image": "vec3"}})

    # A text-only edit, no bindingTypes key at all -- the VEC3 map must still be in force.
    _, notes = s.handle("textDocument/didChange",
                         {"textDocument": {"uri": "u"},
                          "contentChanges": [{"text": _PROG + " // trivial"}]})
    if "E3301" in _error_codes(notes[0]["params"]["diagnostics"]):
        r.ok("text-only didChange (no bindingTypes key) keeps the VEC3 map: E3301 persists")
    else:
        r.fail("HOOK-2 didChange persistence", f"got {notes[0]['params']['diagnostics']}")

    # A bindings-only edit (no contentChanges at all) must ALSO republish -- against the
    # already-stored text -- rather than let diagnostics go stale until the next text edit.
    _, notes2 = s.handle("textDocument/didChange",
                          {"textDocument": {"uri": "u"}, "bindingTypes": {}})
    if notes2 and _error_codes(notes2[0]["params"]["diagnostics"]) == []:
        r.ok("a bindings-only update (dropping the map to {}) republishes immediately, clean")
    else:
        r.fail("HOOK-2 bindings-only update", f"got {notes2!r}")

    # And truly nothing changing (same text, no map key) is still the pre-existing no-op --
    # the new bookkeeping must not make an unrelated fast path fire notifications it didn't
    # fire before.
    _, notes3 = s.handle("textDocument/didChange",
                          {"textDocument": {"uri": "u"},
                           "contentChanges": [{"text": _PROG + " // trivial"}]})
    if notes3 == []:
        r.ok("no actual change (same text, no map key) is still a silent no-op")
    else:
        r.fail("HOOK-2 no-op regression", f"got {notes3!r}")


# ── safety: a malformed map is inert, never fatal ─────────────────────────────

def test_hook2_malformed_map_never_crashes(r: SubTestResult):
    print("\n--- HOOK-2: a malformed bindingTypes value never crashes the server ---")
    s = tex_lsp.LSPServer()
    malformed = [
        "not-a-dict", ["a", "list"], 42, True, None,
        {"image": 12345},                    # non-string value
        {"image": "vec99"},                  # unrecognised type name
        {"image": "VEC3"},                   # wrong case -- no coercion, dropped not crashed
        {7: "vec3"},                         # non-string key
        {"image": "vec3", "bad": object()},  # a value real JSON could never carry
    ]
    crashed = []
    for i, bt in enumerate(malformed):
        uri = f"m{i}"
        try:
            _, notes = s.handle("textDocument/didOpen",
                                 {"textDocument": {"uri": uri, "text": _PROG},
                                  "bindingTypes": bt})
            assert notes and "diagnostics" in notes[0]["params"]
            _, notes = s.handle("textDocument/didChange",
                                 {"textDocument": {"uri": uri},
                                  "contentChanges": [{"text": _PROG + " // x"}],
                                  "bindingTypes": bt})
        except Exception as e:
            crashed.append(f"{bt!r} -> {type(e).__name__}: {e}")
    if not crashed:
        r.ok(f"{len(malformed)} malformed bindingTypes shapes handled without raising")
    else:
        r.fail("HOOK-2 malformed map", "; ".join(crashed))

    # The server must still be alive and answering afterwards -- not just "didn't raise
    # inside this call", but genuinely still serviceable (mirrors test_lsp_bad_frames' proof
    # that a bad request/frame can't tear the session down).
    init, _ = s.handle("initialize", {})
    if "hoverProvider" in init.get("capabilities", {}):
        r.ok("server still answers 'initialize' after every malformed bindingTypes value")
    else:
        r.fail("HOOK-2 server survival", f"got {init!r}")

    # _parse_binding_types' own contract, directly: non-dict -> None (leave the stored map
    # alone); dict -> a filtered dict, never raising, never keeping an unresolvable entry.
    cases_none = ["x", ["a"], 42, True, None, object()]
    cases_dict = [
        ({}, {}),
        ({"a": "vec3"}, {"a": TEXType.VEC3}),
        ({"a": "vec3", "b": "nope"}, {"a": TEXType.VEC3}),
        ({"a": "VEC3"}, {}),
        ({1: "vec3"}, {}),
        ({"a": "array"}, {}),
    ]
    bad = [c for c in cases_none if tex_lsp._parse_binding_types(c) is not None]
    bad += [(raw, want) for raw, want in cases_dict
            if tex_lsp._parse_binding_types(raw) != want]
    if not bad:
        r.ok("_parse_binding_types: non-dict -> None, dict -> filtered dict, never raises")
    else:
        r.fail("HOOK-2 _parse_binding_types", f"diverged from the documented contract: {bad}")
