"""
The user tool store: file-name mapping, the overwrite guard, and reading awkward files.
Every file is built in a temp directory at run time.
"""
import json
import os
import tempfile

import pytest

from TEX_Wrangle import tex_tool as T


def _raw(name="x"):
    return {"name": name, "tex_language": "0.25", "code": "@OUT = vec4(1.0);"}


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as d:
        yield d


# ── file names ───────────────────────────────────────────────────────────────

def test_ordinary_names_keep_their_file_names():
    assert T._safe_tool_filename("My Grade") == "My_Grade.textool"
    assert T._safe_tool_filename("tool") == "tool.textool"
    assert T._safe_tool_filename("a.b-c_d") == "a.b-c_d.textool"


def test_names_with_no_ascii_letters_get_distinct_files():
    names = ["色調", "Ñ", "___"]
    files = {T._safe_tool_filename(n) for n in names}
    assert len(files) == 3
    assert "tool.textool" not in files
    assert T._safe_tool_filename("色調") == T._safe_tool_filename("色調")


@pytest.mark.parametrize("name", ["CON", "nul", "Aux", "PRN", "com1", "LPT9", "con.x"])
def test_windows_device_names_are_not_used_as_file_stems(name):
    f = T._safe_tool_filename(name)
    stem = f.split(".")[0].upper()
    assert stem not in {"CON", "PRN", "AUX", "NUL"} and not stem.startswith(("COM", "LPT")), f


def test_a_very_long_name_stays_under_the_component_limit_and_distinct(store):
    a, b = "n" * 300 + "a", "n" * 300 + "b"
    fa, fb = T._safe_tool_filename(a), T._safe_tool_filename(b)
    assert len(fa) < 200 and fa != fb
    path = T.write_tool(_raw(a), store)
    assert os.path.exists(path) and T.load_tool(path).name == a


# ── the overwrite guard ──────────────────────────────────────────────────────

@pytest.mark.parametrize("junk", [b"[1, 2]", b"\xff\xfe\x00 not utf8", b"not json", b"42", b""])
def test_a_corrupt_file_at_the_target_is_overwritten(store, junk):
    path = os.path.join(store, T._safe_tool_filename("x"))
    with open(path, "wb") as f:
        f.write(junk)
    assert T.write_tool(_raw("x"), store) == path
    assert T.load_tool(path).name == "x"


def test_a_different_tool_at_the_target_is_never_clobbered(store):
    T.write_tool(_raw("My Grade"), store)
    with pytest.raises(T.TEXToolError, match="different tool"):
        T.write_tool(_raw("My/Grade"), store)
    assert T.load_tool(os.path.join(store, "My_Grade.textool")).name == "My Grade"


def test_an_unreadable_target_is_not_treated_as_corrupt(store, monkeypatch):
    path = T.write_tool(_raw("keep"), store)
    real_open = open

    def deny(p, *a, **k):
        if os.path.abspath(str(p)) == os.path.abspath(path) and "r" in (a[0] if a else k.get("mode", "r")):
            raise PermissionError("locked")
        return real_open(p, *a, **k)

    monkeypatch.setattr("builtins.open", deny)
    with pytest.raises(T.TEXToolError):
        T.write_tool(_raw("keep"), store)
    monkeypatch.undo()
    assert T.load_tool(path).name == "keep"


# ── reading ──────────────────────────────────────────────────────────────────

def test_a_tool_saved_with_a_utf8_bom_loads(store):
    path = os.path.join(store, "bom.textool")
    with open(path, "wb") as f:
        f.write(b"\xef\xbb\xbf" + json.dumps(_raw("bom")).encode("utf-8"))
    assert T.load_tool(path).name == "bom"


def test_a_non_regular_file_is_refused_not_opened(store):
    path = os.path.join(store, "dir.textool")
    os.mkdir(path)
    with pytest.raises(T.TEXToolError, match="regular file"):
        T.load_tool(path)
    T._SUMMARY_CACHE.clear()
    (entry,) = T.load_all_tools(store)
    assert entry["name"] == "dir.textool" and "error" in entry
    T._SUMMARY_CACHE.clear()
