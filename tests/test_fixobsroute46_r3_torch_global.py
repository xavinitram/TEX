"""FIX-OBSROUTE (v0.46 Phase C) — R3 [R3#3, low]: `tex_marshalling`'s hot functions resolve
`torch` once, not per call.

LINT-46 already moved `import torch` out of MODULE scope (it must never sit on
`tex_api.check()`'s pure-lint import path — `test_lint46_check_torch_free.py` guards that)
and into each of the ~11 functions that actually touch a tensor. That closed the torch-free
requirement but left `import torch` as a FUNCTION-LOCAL statement, re-executed (one
`IMPORT_NAME` bytecode + a `sys.modules` dict lookup) on every single call — measured at
~35ns/call, small but paid on the marshalling hot path (once per binding per cook).

Fixed with a module-global cache (`tex_marshalling._torch`, resolved by `_torch_mod()`):
the first tensor-touching call pays one real `import torch`, and every later call in the
process reads the cached module object straight off the global — `check()`'s own path still
never touches it, so LINT-46's contract is unaffected (re-checked here too, structurally).
"""
import ast
import builtins
import pathlib

import torch

from TEX_Wrangle import tex_marshalling


_SRC_PATH = pathlib.Path(tex_marshalling.__file__)


def _parse():
    return ast.parse(_SRC_PATH.read_text(encoding="utf-8"))


def test_r3_no_function_local_import_torch_remains(r):
    """Structural proof of the fix's mechanism: no function body in `tex_marshalling.py`
    contains its own `import torch` statement any more (the R3 defect) — every
    tensor-touching function reads the cached module via `_torch_mod()` instead. The
    module-level `if TYPE_CHECKING: import torch` guard, and `_torch_mod()`'s OWN single
    real import, are the only two `import torch` occurrences left in the file."""
    print("\n--- FIX-OBSROUTE R3: no per-call `import torch` remains ---")
    tree = _parse()
    offenders = []

    class _Visitor(ast.NodeVisitor):
        def __init__(self):
            self.func_stack = []

        def _in_torch_mod(self):
            return self.func_stack and self.func_stack[-1] == "_torch_mod"

        def visit_FunctionDef(self, node):
            self.func_stack.append(node.name)
            self.generic_visit(node)
            self.func_stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Import(self, node):
            if self.func_stack and not self._in_torch_mod():
                for alias in node.names:
                    if alias.name == "torch":
                        offenders.append(f"{self.func_stack[-1]} (line {node.lineno})")
            self.generic_visit(node)

    _Visitor().visit(tree)
    if offenders:
        r.fail("FIX-OBSROUTE R3", f"function-local `import torch` still present in: "
               f"{', '.join(offenders)}")
    else:
        r.ok("no function-local `import torch` remains outside `_torch_mod()` itself")


def test_r3_torch_mod_helper_exists_and_caches(r):
    """Behavioural half: `_torch_mod()` exists, returns the real torch module, and — after
    being reset — the SAME object is returned on a second call without re-importing (the
    cache is a module global, not per-call state)."""
    print("\n--- FIX-OBSROUTE R3: _torch_mod() resolves and caches ---")
    if not hasattr(tex_marshalling, "_torch_mod"):
        r.fail("FIX-OBSROUTE R3", "tex_marshalling._torch_mod does not exist")
        return
    saved = tex_marshalling._torch
    try:
        tex_marshalling._torch = None
        first = tex_marshalling._torch_mod()
        if first is not torch:
            r.fail("FIX-OBSROUTE R3", "_torch_mod() did not return the real torch module")
            return
        if tex_marshalling._torch is not torch:
            r.fail("FIX-OBSROUTE R3", "_torch_mod() did not cache its result on the module "
                   "global `_torch`")
            return
        r.ok("_torch_mod() resolves to the real torch module and caches it on `_torch`")

        second = tex_marshalling._torch_mod()
        if second is first:
            r.ok("a second call returns the identical cached object")
        else:
            r.fail("FIX-OBSROUTE R3", "a second call returned a different object")
    finally:
        tex_marshalling._torch = saved


def test_r3_tensor_touching_functions_still_work(r):
    """Non-regression: functions that used to `import torch` locally still behave correctly
    after being switched to `torch = _torch_mod()`."""
    print("\n--- FIX-OBSROUTE R3: tensor-touching functions still behave correctly ---")
    saved = tex_marshalling._torch
    try:
        tex_marshalling._torch = None   # force the cache-miss path on the very first call
        img_int = torch.zeros(1, 4, 4, 3, dtype=torch.uint8)
        out = tex_marshalling.to_fp32_if_int_image(img_int)
        if out.dtype != torch.float32:
            r.fail("FIX-OBSROUTE R3 to_fp32_if_int_image", f"expected float32, got {out.dtype}")
        else:
            r.ok("to_fp32_if_int_image still casts an int image to fp32")

        t = torch.rand(4, 4, 4, 4)
        fp = tex_marshalling.tensor_fingerprint(t)
        if not isinstance(fp, str) or not fp:
            r.fail("FIX-OBSROUTE R3 tensor_fingerprint", f"expected a non-empty string, got {fp!r}")
        else:
            r.ok("tensor_fingerprint still returns a fingerprint string")
    finally:
        tex_marshalling._torch = saved


def test_r3_check_still_never_imports_torch(r):
    """The other half of the contract this fix must not disturb: caching `torch` on a
    module global must not make `tex_api.check()`'s pure-lint path reach it — that end-to-
    end, fresh-subprocess proof lives in LINT-46's own
    `test_lint46_check_torch_free.py::test_lint46_tex_api_check_stays_torch_free`, which
    this fix does not touch. This test just confirms `infer_binding_type` (a function this
    fix left untouched — it already resolved torch unconditionally, for its
    `isinstance(value, torch.Tensor)` checks, before and after R3) still behaves correctly
    against the module-global cache."""
    print("\n--- FIX-OBSROUTE R3: check()'s own torch-free contract is undisturbed ---")
    saved = tex_marshalling._torch
    try:
        tex_marshalling._torch = None
        from TEX_Wrangle.tex_compiler.types import TEXType
        t = tex_marshalling.infer_binding_type(3.0)
        if t != TEXType.FLOAT:
            r.fail("FIX-OBSROUTE R3 infer_binding_type", f"expected FLOAT for a bare float, got {t}")
        else:
            r.ok("infer_binding_type(3.0) still resolves FLOAT against the module-global cache")
    finally:
        tex_marshalling._torch = saved


def test_r3_cached_lookup_performs_no_reimport(r):
    """Deterministic replacement for a former wall-clock microbenchmark (CI-461): the old
    version compared `_torch_mod()` post-warm against a bare `import torch` statement on a
    generous 2x bound and went red in CI on Python 3.10/3.11/3.12 alike, all under
    `pytest --cov` — coverage's line tracer taxes the extra Python-level function call
    `_torch_mod()` makes far more than it taxes a single `IMPORT_NAME` bytecode, so the
    "not slower" claim inverts under tracing (replayed locally with a no-op `sys.settrace`
    tracer standing in for coverage.py: the cached path went from ~2x FASTER than a bare
    import untraced to ~3x SLOWER than it while traced). A wall-clock bound can't tell
    "the cache isn't paying for itself" apart from "a tracer is running"; the actual
    contract can: after the first call warms the cache, later calls import nothing at all.
    """
    print("\n--- FIX-OBSROUTE R3: cached lookup performs no re-import ---")
    saved = tex_marshalling._torch
    try:
        tex_marshalling._torch = None
        tex_marshalling._torch_mod()   # warm the cache — the one call allowed to import

        real_import = builtins.__import__
        import_count = 0

        def _counting_import(name, *args, **kwargs):
            nonlocal import_count
            if name == "torch":
                import_count += 1
            return real_import(name, *args, **kwargs)

        last = None
        builtins.__import__ = _counting_import
        try:
            for _ in range(500):
                last = tex_marshalling._torch_mod()
        finally:
            builtins.__import__ = real_import

        if import_count != 0:
            r.fail("FIX-OBSROUTE R3 microbenchmark", f"_torch_mod() re-imported torch "
                   f"{import_count} time(s) across 500 post-warm calls — the cache is not "
                   f"doing its job")
        elif last is not torch:
            r.fail("FIX-OBSROUTE R3 microbenchmark", "a post-warm _torch_mod() call did not "
                   "return the real torch module")
        else:
            r.ok("500 post-warm _torch_mod() calls performed 0 imports and returned the "
                 "cached module")
    finally:
        tex_marshalling._torch = saved
