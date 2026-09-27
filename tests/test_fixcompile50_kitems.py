"""FIX-COMPILE (v0.50.0 Phase C) — K1-K6, red-first per item. Each test below
reproduces its finding against the pre-fix shape directly (a local monkeypatch/
stand-in reconstructing the old behaviour, or a direct exercise of the real fix)
rather than requiring a checkout of the base commit, so the file is its own
red-first evidence.

PORTABILITY: CPU-only (no CUDA/MSVC/Triton assumed); K1 uses a real `torch.compile(...,
backend="aot_eager")` (no Inductor/toolchain needed for aot_eager) — the same backend
COMPILETRY-50's own D2 test uses for the identical reason.
"""
import threading
import time
import types

import torch
import torch._dynamo.config as _dynamo_config

from helpers import *  # noqa: F401,F403
from TEX_Wrangle.tex_runtime import codegen_persist as CP
from TEX_Wrangle.tex_runtime import fncalls_compile as FC
from TEX_Wrangle.tex_runtime import compiled as C


# ── K1: codegen's exec-globals lacked __name__ -> Dynamo KeyError on a graph-break
# resume. Fix: a REAL per-build module registered in sys.modules (_codegen_exec_namespace),
# not a bare dict and not a synthetic placeholder string. ──────────────────────────────

_K1_HELPER_SRC = """
import torch as _torch

@_torch.compiler.disable()
def _helper(x):
    return x + 1.0
"""

_K1_FN_SRC = """
def _tex_fn(x):
    y = x * 2.0
    y = _helper(y)
    z = y * 3.0
    return z
"""


def _k1_build(namespace: dict):
    exec(compile(_K1_HELPER_SRC, "<k1-helper>", "exec"), namespace)
    exec(compile(_K1_FN_SRC, "<k1-fn>", "exec"), namespace)
    return namespace["_tex_fn"]


def test_k1_bare_dict_globals_raise_keyerror_on_resume(r: SubTestResult):
    """RED-FIRST EVIDENCE: the exact pre-fix shape (a bare dict with no dunders, the
    literal `{"_MF": ..., "_CK": ..., "_SCM": ...}` `codegen.py`/`codegen_persist.py`
    used to hand `exec()`) really does raise `KeyError: '__name__'` inside Dynamo's
    graph-break resume, under `caching_precompile=True` -- the exact flag
    `compiled._precompile_ctx()` sets around every real compile attempt."""
    print("\n--- K1 (red-first): a bare exec-globals dict raises on a graph-break resume ---")
    fn = _k1_build({})
    try:
        with _dynamo_config.patch(caching_precompile=True):
            cfn = torch.compile(fn, backend="aot_eager", fullgraph=False)
            cfn(torch.ones(3))
        r.fail("K1 red-first", "expected a KeyError on '__name__' from the bare-dict "
               "globals shape, but the call succeeded -- the red-first repro no longer "
               "reproduces the defect this fix targets")
    except Exception as e:
        if "__name__" in str(e) or "__name__" in type(e).__name__:
            r.ok(f"K1 red-first confirmed: bare-dict globals raise "
                 f"{type(e).__name__} mentioning __name__, exactly B3#1's finding")
        else:
            r.fail("K1 red-first", f"raised {type(e).__name__}: {e!r} -- not the "
                   "expected __name__ KeyError shape")
    finally:
        torch._dynamo.reset()


def test_k1_codegen_exec_namespace_survives_a_graph_break_resume(r: SubTestResult):
    """GREEN: the real fix, `codegen_persist._codegen_exec_namespace`, used exactly the
    way `codegen.py`'s `_CodeGen.build()` now uses it -- a real module in sys.modules,
    not a bare dict and not a synthetic unregistered string (B3#1's own middle finding:
    that trades the KeyError for ModuleNotFoundError)."""
    print("\n--- K1: _codegen_exec_namespace's real module survives a graph-break resume ---")
    filename = "<tex_codegen_k1testfixture>"
    try:
        namespace = CP._codegen_exec_namespace(filename, {})
        mod_name = CP._codegen_module_name(filename)
        assert mod_name in __import__("sys").modules, (
            "the fix must register a REAL module in sys.modules, found none")
        assert __import__("sys").modules[mod_name].__dict__ is namespace, (
            "the returned namespace must BE the registered module's own __dict__")
        fn = _k1_build(namespace)
        with _dynamo_config.patch(caching_precompile=True):
            cfn = torch.compile(fn, backend="aot_eager", fullgraph=False)
            out1 = cfn(torch.ones(3))
            out2 = cfn(torch.ones(3) * 2)   # a second call, same guard -- exercises resume reuse
        assert torch.equal(out1, torch.tensor([9.0, 9.0, 9.0]))
        assert torch.equal(out2, torch.tensor([15.0, 15.0, 15.0]))
        r.ok("K1: a real per-build module survives torch.compile's graph-break resume "
             "under caching_precompile, with correct results")
    except Exception as e:
        r.fail("K1 codegen_exec_namespace", f"{type(e).__name__}: {e}")
    finally:
        torch._dynamo.reset()
        __import__("sys").modules.pop(CP._codegen_module_name(filename), None)
        import linecache
        linecache.cache.pop(filename, None)
        try:
            CP._LINECACHE_KEYS.remove(filename)
        except ValueError:
            pass


def test_k1_module_is_bounded_and_evicted_with_its_linecache_entry(r: SubTestResult):
    """K1's own bound: the synthetic module is evicted from sys.modules in lockstep with
    its paired linecache entry, so a long session cannot leak one module per fingerprint
    forever (B3#1's own stated risk for the 'point at a real, shared module' alternative
    this fix deliberately avoids)."""
    print("\n--- K1: the synthetic module is evicted alongside its linecache entry ---")
    import sys as _sys
    saved_keys = list(CP._LINECACHE_KEYS)
    saved_max = CP._LINECACHE_MAX
    CP._LINECACHE_MAX = 2
    names = [f"<tex_codegen_k1bound{i}>" for i in range(4)]
    try:
        for n in names:
            CP._register_codegen_linecache(n, "def _tex_fn():\n    return 1\n")
            CP._codegen_exec_namespace(n, {})
        # only the last _LINECACHE_MAX filenames should still have BOTH a linecache
        # entry and a registered module -- the earlier ones must have been evicted
        # from both together.
        still_present = [n for n in names if n in CP._LINECACHE_KEYS]
        assert len(still_present) == CP._LINECACHE_MAX, still_present
        for n in names:
            mod_name = CP._codegen_module_name(n)
            in_linecache = n in __import__("linecache").cache
            in_modules = mod_name in _sys.modules
            assert in_linecache == in_modules, (
                f"{n}: linecache present={in_linecache} but sys.modules present="
                f"{in_modules} -- the two registries drifted apart")
        r.ok(f"K1: linecache and sys.modules stay in lockstep across eviction "
             f"({len(names)} builds, cap {CP._LINECACHE_MAX})")
    except Exception as e:
        r.fail("K1 bounded eviction", f"{type(e).__name__}: {e}")
    finally:
        for n in names:
            _sys.modules.pop(CP._codegen_module_name(n), None)
            __import__("linecache").cache.pop(n, None)
        CP._LINECACHE_MAX = saved_max
        CP._LINECACHE_KEYS.clear()
        CP._LINECACHE_KEYS.extend(saved_keys)
