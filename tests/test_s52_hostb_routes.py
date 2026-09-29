"""v0.52 sweep: package-root route rows (event-loop offload, free_caches contract, snippet
cache after a transient read failure)."""
import builtins
import os

import pytest

import TEX_Wrangle
from test_v035_hygiene import _route_probe


_OFFLOAD_BODY = '''
import threading
import TEX_Wrangle.tex_fusion, TEX_Wrangle.tex_tool

seen = {}

def _spy(name, result):
    def f(*a, **k):
        seen[name] = threading.current_thread() is threading.main_thread()
        return result
    return f

TEX_Wrangle.tex_fusion.preflight_from_spec = _spy("preflight", {"ok": True, "error": None,
                                                                 "stage_of_error": None, "stats": None})
TEX_Wrangle.tex_tool.load_all_tools = _spy("tools", [])
drive("POST", "/tex_wrangle/chain_preflight", body={"stages": [], "terminal_code": ""})
drive("GET", "/tex_wrangle/list_tools")
OUT["on_loop_thread"] = seen
'''


def test_preflight_and_tool_listing_run_off_the_event_loop():
    out = _route_probe(_OFFLOAD_BODY)
    assert out["on_loop_thread"] == {"preflight": False, "tools": False}


_FREE_BODY = '''
import types
import TEX_Wrangle.tex_memory

calls = []
TEX_Wrangle.tex_memory.free_tensor_caches = lambda: calls.append("swept")

_stub.PromptServer.instance.prompt_queue = types.SimpleNamespace(
    get_current_queue_volatile=lambda: ([("running-item",)], []))
res = drive("POST", "/tex_wrangle/free_caches")
OUT["busy"] = [res.status, res.body["ok"], list(calls)]

_stub.PromptServer.instance.prompt_queue = types.SimpleNamespace(
    get_current_queue_volatile=lambda: ([], []))
res = drive("POST", "/tex_wrangle/free_caches")
OUT["idle"] = [res.status, res.body["ok"], list(calls)]
'''


def test_free_caches_refuses_while_a_cook_is_running():
    out = _route_probe(_FREE_BODY)
    assert out["busy"] == [409, False, []]
    assert out["idle"][0] == 200 and out["idle"][2] == ["swept"]


def test_a_transient_read_failure_does_not_cache_a_partial_snippet_set(tmp_path, monkeypatch):
    (tmp_path / "a_one.tex").write_text("@OUT = @A;", encoding="utf-8")
    (tmp_path / "b_two.tex").write_text("@OUT = @A * 2.0;", encoding="utf-8")
    monkeypatch.setattr(TEX_Wrangle, "_EXAMPLES_DIR", str(tmp_path))
    monkeypatch.setattr(TEX_Wrangle, "_snippets_cache", None)
    real_open = builtins.open
    locked = {"on": True}

    def flaky(path, *a, **k):
        if locked["on"] and os.path.basename(str(path)) == "b_two.tex":
            raise PermissionError("locked by another process")
        return real_open(path, *a, **k)

    monkeypatch.setattr(builtins, "open", flaky)
    first = TEX_Wrangle._load_example_snippets()
    assert len(first) == 1
    assert TEX_Wrangle._snippets_cache is None
    locked["on"] = False
    second = TEX_Wrangle._load_example_snippets()
    assert len(second) == 2 and TEX_Wrangle._snippets_cache is second
