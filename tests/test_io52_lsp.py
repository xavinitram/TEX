"""
Language server: the advertised pull-diagnostics request is answered, and positions are UTF-16
code units (LSP's default), so a non-BMP character earlier on the line does not shift them.
"""
from TEX_Wrangle import tex_lsp as L

_URI = "file:///t.tex"
_SRC = 'string s = "\U0001F600"; @OUT = vec4(zzz);'      # the emoji is 2 UTF-16 units, 1 code point


def _open(src=_SRC):
    srv = L.LSPServer()
    srv.handle("textDocument/didOpen", {"textDocument": {"uri": _URI, "text": src}})
    return srv


def test_pull_diagnostics_is_advertised_and_answered_with_a_full_report():
    srv = L.LSPServer()
    caps = srv.handle("initialize", {})[0]["capabilities"]
    assert "diagnosticProvider" in caps
    srv = _open()
    result, notes = srv.handle("textDocument/diagnostic", {"textDocument": {"uri": _URI}})
    assert notes == []
    assert result["kind"] == "full"
    assert result["items"] == L.diagnostics_for(_SRC)
    assert any("zzz" in d["message"] for d in result["items"])


def test_pull_diagnostics_for_an_unknown_document_is_an_empty_report():
    result, _ = L.LSPServer().handle("textDocument/diagnostic", {"textDocument": {"uri": "x"}})
    assert result == {"kind": "full", "items": []}


def test_pull_diagnostics_use_the_documents_binding_map():
    srv = L.LSPServer()
    src = "@OUT = vec4(@A.a);"
    srv.handle("textDocument/didOpen", {"textDocument": {"uri": _URI, "text": src},
                                        "bindingTypes": {"A": "vec3"}})
    pulled = srv.handle("textDocument/diagnostic", {"textDocument": {"uri": _URI}})[0]["items"]
    assert pulled == L.diagnostics_for(src, srv.binding_types[_URI])
    assert pulled != L.diagnostics_for(src)


def test_diagnostic_columns_are_utf16_units():
    (d,) = [d for d in L.diagnostics_for(_SRC) if "zzz" in d["message"]]
    assert d["range"]["start"] == {"line": 0, "character": _SRC.index("zzz") + 1}


def test_hover_reads_a_utf16_position():
    line = "/*" + "😀" * 6 + " sin(1.0);"
    units = len(("/*" + "😀" * 6 + " ").encode("utf-16-le")) // 2   # column of `s` in UTF-16
    assert units == 15 and line.index("sin") == 9
    for col in (units, units + 2):                                        # both inside `sin`
        hov = L.hover_for(line, 0, col)
        assert hov is not None and "sin" in hov["contents"]["value"]
    assert L.hover_for("@OUT = vec4(sin(1.0));", 0, 14) is not None       # ASCII unchanged
