"""Robustness of the tool palette, the snippet store, the touched-tier selection and the MSVC
environment parse: each row fails on the unfixed code."""
import json
import os
import sys
import tempfile

from helpers import SubTestResult


def test_palette_survives_hostile_textool_files(r: SubTestResult):
    print("\n--- palette: a bad .textool is an error entry, never an abort ---")
    from TEX_Wrangle import tex_tool
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "a_binary.textool"), "wb") as f:
            f.write(b"\xff\xfe\x00\x80not utf8")
        with open(os.path.join(d, "b_deep.textool"), "w") as f:
            f.write("[" * 200000)
        with open(os.path.join(d, "c_ok.textool"), "w") as f:
            f.write("{}")
        tex_tool._SUMMARY_CACHE.clear()
        try:
            out = tex_tool.load_all_tools(d)
        except Exception as e:
            r.fail("palette listing", f"raised {type(e).__name__}: {e}")
            return
        names = sorted(o["name"] for o in out)
        if names == ["a_binary.textool", "b_deep.textool", "c_ok.textool"] \
                and all("error" in o for o in out):
            r.ok("three bad files -> three error entries")
        else:
            r.fail("palette entries", repr(out))

        # A file that vanishes between listing and load.
        real = tex_tool.list_tools
        tex_tool.list_tools = lambda dir=None: [os.path.join(d, "gone.textool")]
        try:
            out = tex_tool.load_all_tools(d)
            if len(out) == 1 and "error" in out[0]:
                r.ok("vanished file -> error entry")
            else:
                r.fail("vanished file", repr(out))
        except Exception as e:
            r.fail("vanished file", f"raised {type(e).__name__}: {e}")
        finally:
            tex_tool.list_tools = real
        tex_tool._SUMMARY_CACHE.clear()


def test_snippets_save_survives_lone_surrogate(r: SubTestResult):
    print("\n--- snippets: a lone surrogate does not fail every save ---")
    from TEX_Wrangle import tex_snippets
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "tex_wrangle", "user_snippets.json")
        real = tex_snippets._snippets_path
        tex_snippets._snippets_path = lambda: p
        try:
            ok = tex_snippets.save_user_snippets({"a": "x\ud800y", "b": "caf\u00e9"})
            if not ok:
                r.fail("save with surrogate", "save_user_snippets returned False")
                return
            with open(p, "rb") as f:
                data = json.loads(f.read().decode("utf-8"))
            if data["a"] == "x\ud800y" and data["b"] == "caf\u00e9":
                r.ok("surrogate round-trips")
            else:
                r.fail("round trip", repr(data))
            # The common case keeps its UTF-8 bytes.
            tex_snippets.save_user_snippets({"b": "caf\u00e9"})
            with open(p, "rb") as f:
                if "caf\u00e9".encode("utf-8") in f.read():
                    r.ok("plain non-ASCII stays UTF-8")
                else:
                    r.fail("plain non-ASCII", "was escaped")
        finally:
            tex_snippets._snippets_path = real


def test_touched_selection_reads_from_package_import_module(r: SubTestResult):
    print("\n--- gate: `from TEX_Wrangle.pkg import mod` selects on pkg.mod ---")
    import importlib.util
    import pathlib
    path = pathlib.Path(__file__).resolve().parent.parent / "tools" / "gate.py"
    spec = importlib.util.spec_from_file_location("_misc52_gate", str(path))
    g = importlib.util.module_from_spec(spec)
    sys.modules["_misc52_gate"] = g
    spec.loader.exec_module(g)
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "test_x.py")
        with open(f, "w") as fh:
            fh.write("from TEX_Wrangle.tex_runtime import compiled\n"
                     "S = 'from TEX_Wrangle.tex_compiler import optimizer'\n")
        refs = g._test_module_refs(f)
        for want in ("tex_runtime.compiled", "tex_compiler.optimizer"):
            if want in refs:
                r.ok(f"{want} resolved")
            else:
                r.fail(want, f"refs={sorted(refs)}")
    real = g._importable_as_tex_wrangle
    g._importable_as_tex_wrangle = lambda: False
    try:
        rc = g.main(["--tier", "cheap"])
    finally:
        g._importable_as_tex_wrangle = real
    if rc == 1:
        r.ok("not-importable refusal is RED (1)")
    else:
        r.fail("refusal rc", f"got {rc}, want 1")


def test_msvc_env_dump_upcases_names(r: SubTestResult):
    print("\n--- compiled: MSVC env dump keys are upper-cased ---")
    from TEX_Wrangle.tex_runtime import compiled
    env = compiled._parse_env_dump("Path=C:\\cl\nINCLUDE=I\nweird=a=b\nnoequals\n")
    if env.get("PATH") == "C:\\cl" and env.get("INCLUDE") == "I" and env.get("WEIRD") == "a=b":
        r.ok("Path -> PATH")
    else:
        r.fail("env parse", repr(env))
