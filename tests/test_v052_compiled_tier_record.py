"""The compiled tiers record their SUCCESS tier on the cook thread, and the cancel-aware
codegen memo never keys on a recycled object id."""
import torch

from helpers import *  # noqa: F401,F403
from TEX_Wrangle.tex_runtime import compiled as C
from TEX_Wrangle.tex_runtime import tier_trace


def _setup():
    from TEX_Wrangle.tex_cache import parse_and_split
    from TEX_Wrangle.tex_compiler.type_checker import TypeChecker
    from TEX_Wrangle.tex_runtime.interpreter import _collect_identifiers
    code = "vec3 c=@A.rgb*1.3 - 0.1; c=clamp(c,0.0,1.0); @OUT=vec4(c,1.0);"
    bt = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}
    prog = parse_and_split(code, bt)
    tm = TypeChecker(binding_types=bt, source=code).check(prog)
    return prog, tm, _collect_identifiers(prog)


def _fake_compiled(program, bindings, type_map, device, lcc, names, scale=None):
    return {n: torch.zeros(1, 8, 8, 4) for n in (names or ["OUT"])}


def test_execute_compiled_records_torch_compile_on_the_cook_thread(monkeypatch):
    prog, tm, used = _setup()
    fp = "fu52_exec_fp"
    cache_key = (fp, "cpu", "fp32")
    monkeypatch.setattr(C, "_try_compile", lambda *a, **k: (_fake_compiled, "inductor"))
    monkeypatch.setattr(C, "_COMPILE_OP_THRESHOLD", 0)
    try:
        tier_trace.reset()
        C.execute_compiled(prog, {"A": torch.rand(1, 8, 8, 3)}, tm, "cpu", fp,
                           output_names=["OUT"], used_builtins=used)
        rec = tier_trace.last()
        assert rec is not None and rec.tier == "torch_compile", rec
        assert rec.fallback_from is None, rec
        tier_trace.reset()          # and the warm (cached) path records it as well
        C.execute_compiled(prog, {"A": torch.rand(1, 8, 8, 3)}, tm, "cpu", fp,
                           output_names=["OUT"], used_builtins=used)
        assert tier_trace.last().tier == "torch_compile", tier_trace.last()
    finally:
        C._compiled_cache.pop(cache_key, None)
        C._verify_state.pop(cache_key, None)


def test_execute_compiled_without_a_backend_does_not_claim_torch_compile(monkeypatch):
    prog, tm, used = _setup()
    fp = "fu52_nobackend_fp"
    monkeypatch.setattr(C, "_try_compile", lambda *a, **k: None)
    monkeypatch.setattr(C, "_COMPILE_OP_THRESHOLD", 0)
    tier_trace.reset()
    C.execute_compiled(prog, {"A": torch.rand(1, 8, 8, 3)}, tm, "cpu", fp,
                       output_names=["OUT"], used_builtins=used)
    rec = tier_trace.last()
    assert rec is None or rec.tier != "torch_compile", rec


def test_a_committed_auto_cook_records_torch_compile(monkeypatch):
    from TEX_Wrangle.tex_runtime import autotier as AT
    prog, tm, used = _setup()
    img = torch.rand(1, 8, 8, 3)
    fp = "fu52_auto_fp"
    cache_key = (fp, "cpu", "fp32")
    monkeypatch.setattr(C, "_try_compile", lambda *a, **k: (_fake_compiled, "inductor"))
    monkeypatch.setattr(C, "compile_capability_async",
                        lambda: {"cuda_inductor": True, "cpu_inductor": True, "reason": {}})
    AT.reset()
    try:
        key = AT.make_key(fp, "cpu", "fp32", C._consensus_extent({"A": img}, prog))
        for _ in range(40):
            C.run_auto(prog, {"A": img}, tm, "cpu", fp, output_names=["OUT"], used_builtins=used)
            C._drain_bg_for_test()
            if AT.verdict(key) == AT.COMMITTED:
                break
        assert AT.verdict(key) == AT.COMMITTED, AT.verdict(key)
        tier_trace.reset()
        C.run_auto(prog, {"A": img}, tm, "cpu", fp, output_names=["OUT"], used_builtins=used)
        assert tier_trace.last() is not None and tier_trace.last().tier == "torch_compile", \
            tier_trace.last()
    finally:
        C._drain_bg_for_test()
        C._compiled_cache.pop(cache_key, None)
        AT.reset()


def test_the_cancel_codegen_memo_is_not_keyed_on_an_object_id(monkeypatch):
    prog, tm, _ = _setup()
    calls = []
    monkeypatch.setattr(C, "_try_codegen", lambda *a, **k: calls.append(1) or (lambda: None))
    saved = dict(C._cancel_codegen_memo)
    C._cancel_codegen_memo.clear()
    try:
        C._get_or_make_cancel_codegen_fn(prog, tm, None)
        C._get_or_make_cancel_codegen_fn(prog, tm, None)
        assert len(calls) == 2 and not C._cancel_codegen_memo, (calls, C._cancel_codegen_memo)
        C._get_or_make_cancel_codegen_fn(prog, tm, "fu52_cancel_fp")
        C._get_or_make_cancel_codegen_fn(prog, tm, "fu52_cancel_fp")
        assert len(calls) == 3, "a fingerprinted build must still be memoized"
    finally:
        C._cancel_codegen_memo.clear()
        C._cancel_codegen_memo.update(saved)
