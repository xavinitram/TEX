"""
The small generators and scanners under tools/: each fixed behaviour has a row that fails on the
unfixed code. Fixtures are built in a temp directory or in memory.
"""
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

_PKG = Path(__file__).resolve().parent.parent
_TOOLS = _PKG / "tools"


def _load(name):
    key = f"_io52_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, str(_TOOLS / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


# ── check_citations ──────────────────────────────────────────────────────────

def _cite(doc_lines, target_source):
    cc = _load("check_citations")
    cc._SPAN_CACHE.clear()
    cc._LINES_CACHE.clear()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "docs").mkdir()
        (root / "mod.py").write_text(target_source, encoding="utf-8", newline="\n")
        (root / "docs" / "d.md").write_text("\n".join(doc_lines) + "\n", encoding="utf-8")
        cits, stats = cc.check(str(root))
    cc._SPAN_CACHE.clear()
    cc._LINES_CACHE.clear()
    return {c.text: c.verdict for c in cits}, stats


_TARGET = ("import os\n"                         # 1
           "\n"                                  # 2
           "@staticmethod\n"                     # 3
           "def widget(n):\n"                    # 4
           "    return n\n")                     # 5


def test_a_citation_on_a_decorator_line_names_the_decorated_symbol():
    got, _ = _cite(["`widget` starts here (`mod.py:3`)."], _TARGET)
    assert got["mod.py:3"] == "ok"


def test_line_zero_and_an_inverted_range_are_dead_citations():
    got, _ = _cite(["Zero (`mod.py:0`).", "Backwards (`mod.py:5-4`)."], _TARGET)
    assert got["mod.py:0"] == "error" and got["mod.py:5-4"] == "error"


def test_exotic_line_separators_do_not_shift_line_numbers():
    src = 'A = "x\x0cy"\nB = "p q"\n\ndef widget():\n    return 1\n'
    got, _ = _cite(["`widget` is here (`mod.py:4`)."], src)
    assert got["mod.py:4"] == "ok"


def test_the_doc_count_is_reported_by_the_check():
    _, stats = _cite(["nothing cited"], _TARGET)
    assert stats["docs"] >= 1


# ── display8 ─────────────────────────────────────────────────────────────────

def test_night_plate_lights_land_inside_a_tiny_frame():
    torch = pytest.importorskip("torch")
    d8 = _load("display8")
    plate = d8.plate_night(6, 6, torch.device("cpu"))
    assert plate.shape == (1, 3, 6, 6)
    assert float(plate.max()) == pytest.approx(20.0)          # the lights are drawn at (0, 0)


# ── gen_error_codes ──────────────────────────────────────────────────────────

def test_error_code_harvest_ignores_a_checkout_path_that_contains_tools(tmp_path):
    gen = _load("gen_error_codes")
    pkg = tmp_path / "tools" / "TEX_Wrangle"                    # the checkout itself lives under 'tools'
    (pkg / "tex_compiler").mkdir(parents=True)
    (pkg / "tests").mkdir()
    (pkg / "tex_compiler" / "x.py").write_text('code = "E1234"\n', encoding="utf-8")
    (pkg / "tests" / "t.py").write_text('code = "E9999"\n', encoding="utf-8")
    old = gen._PKG
    gen._PKG = str(pkg)
    try:
        assert gen.harvest_codes() == ["E1234"]
    finally:
        gen._PKG = old


def test_a_code_in_an_unknown_family_is_refused():
    gen = _load("gen_error_codes")
    with pytest.raises(SystemExit):
        gen.render(["E1234", "E8001"])


# ── scan_commit_messages ─────────────────────────────────────────────────────

def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True,
                          encoding="utf-8", check=True).stdout


def test_a_bad_revision_range_is_exit_two_not_a_traceback(tmp_path, capsys, monkeypatch):
    scm = _load("scan_commit_messages")
    _git(tmp_path, "init", "-q")
    monkeypatch.chdir(tmp_path)
    assert scm.main(["no-such-rev..also-missing"]) == 2
    assert "could not read" in capsys.readouterr().out


def test_git_log_is_decoded_as_utf8_whatever_the_locale(tmp_path, monkeypatch):
    scm = _load("scan_commit_messages")
    _git(tmp_path, "init", "-q")
    (tmp_path / "f.txt").write_text("x", encoding="utf-8")
    _git(tmp_path, "add", "f.txt")
    msg = "café — Ý\n"
    _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q",
         "-m", msg)
    seen = {}
    real = subprocess.run

    def spy(*a, **k):
        seen.update(k)
        return real(*a, **k)

    monkeypatch.setattr(scm.subprocess, "run", spy)
    ((sha, body),) = scm._git_log_messages("HEAD", tmp_path)
    assert seen.get("encoding") == "utf-8"
    assert "café — Ý" in body


# ── the generators run on their own ──────────────────────────────────────────

def test_examples_index_check_passes_run_standalone():
    proc = subprocess.run([sys.executable, "-X", "utf8", str(_TOOLS / "gen_examples_index.py"),
                           "--check"], capture_output=True, text=True, encoding="utf-8",
                          cwd=str(_PKG.parent), env=dict(os.environ, PYTHONPATH=str(_PKG.parent)))
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_llm_cheatsheet_check_is_not_a_failure_without_a_wiki_checkout(tmp_path, monkeypatch, capsys):
    gen = _load("gen_llm_cheatsheet")
    monkeypatch.setattr(gen, "_OUT", str(tmp_path / "wiki" / "LLM-Cheatsheet.md"))
    monkeypatch.setattr(sys, "argv", ["gen_llm_cheatsheet.py", "--check"])
    assert gen.main() == 0
    assert "not checked" in capsys.readouterr().out


def test_cheatsheet_teaches_what_the_engine_does():
    gen = _load("gen_llm_cheatsheet")
    text = gen.render()
    assert ".wzyx" in text and "not" in text.split(".wzyx")[1][:60]
    assert "ix/(iw-1)" in text
    assert "1/(iw-1)" in text
    assert "gives `NaN`" not in text
    for _intent, code in gen.WORKED:
        assert "/ iw," not in code
