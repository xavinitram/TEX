"""
v0.42 HOSTAUDIT-1 — pin the cold-import module count.

Measured in this session (fresh interpreter per row, `sys.path` pointed at a worktree
so a lane's own copy resolves, never the installed tree — per `docs/brief-conventions.md`
"measure in a worktree"):

    bare `import TEX_Wrangle`             ->    1 TEX_Wrangle.* module,   0 torch.* modules
    `from TEX_Wrangle import tex_engine`  ->   44 TEX_Wrangle.* modules, 730 torch.* modules

PORT-6 (v0.35.0) already closed the headline finding the profile audit named
(`TEX_Wrangle/__init__.py:12` used to do `from .tex_node import TEXWrangleNode` at module
scope, so ANY touch of the package loaded the ComfyUI adapter and, behind it, torch and
29 more TEX submodules). The `__init__.py` `__getattr__` it shipped means a bare package
touch now costs exactly what it looks like it costs: nothing but the package's own root.

This is the ratchet for that state, plus the honest floor above it: `tex_engine` itself
IS the tensor engine, so its 730 torch submodules are not avoidable by any import
reordering — `interpreter.py`, `stdlib.py` and friends need `torch.Tensor` to exist. The
TEX-module count is free to move DOWN (a future lane deferring, say, the codegen family
the way PORT-6 deferred the adapter) but a regression that makes bare `import TEX_Wrangle`
pull in torch again — the exact defect PORT-6 fixed — must fail loudly, which is what the
first assertion below exists for.

Both checks run in a SUBPROCESS: this suite has already imported everything, so only a
fresh interpreter can see what a first touch costs.
"""
import pathlib
import subprocess
import sys as _sys

# Ratchets: TEX_MODS may only move DOWN (a future laziness win moves it), never up
# (an eager import creeping back in). TORCH_MODS on the bare touch is the PORT-6
# invariant itself and is pinned at the exact value, not a ceiling.
# SPLIT-I (v0.44 Phase A1): `interpreter.py`'s mechanical split added three sibling
# modules (`interpreter_spatial.py` / `interpreter_control_flow.py` /
# `interpreter_binding.py`) that `Interpreter` composes as mixins, so all three must
# still be imported eagerly at class-definition time — the same reason `masked_flow.py`
# was already eager before this split. 44 -> 47, +1 per new module, nothing else moved.
_BARE_TOUCH_TEX_MODULES_MAX = 1
_BARE_TOUCH_TORCH_MODULES = 0
_TEX_ENGINE_TEX_MODULES_MAX = 47


def _measure(import_stmt: str, custom_nodes: str) -> dict:
    code = (
        "import sys\n"
        f"sys.path.insert(0, {custom_nodes!r})\n"
        "before = set(sys.modules)\n"
        f"{import_stmt}\n"
        "after = set(sys.modules)\n"
        "new = after - before\n"
        "tex = [m for m in new if m.startswith('TEX_Wrangle')]\n"
        "torchm = [m for m in new if m == 'torch' or m.startswith('torch.')]\n"
        "print('TEX', len(tex))\n"
        "print('TORCH', len(torchm))\n"
    )
    proc = subprocess.run([_sys.executable, "-X", "utf8", "-c", code],
                          capture_output=True, text=True, timeout=180)
    if proc.returncode != 0:
        raise RuntimeError(f"subprocess exit {proc.returncode}: {(proc.stderr or '')[-600:]}")
    out = dict(line.split(" ", 1) for line in proc.stdout.strip().splitlines())
    return {"tex": int(out["TEX"]), "torch": int(out["TORCH"])}


def test_hostaudit1_bare_package_touch_stays_torch_free(r):
    """PORT-6's own invariant, re-pinned from this ask: touching the package root alone
    (`import TEX_Wrangle`, never `tex_engine`/`tex_api`/`tex_node`) must not load torch."""
    import TEX_Wrangle
    custom_nodes = str(pathlib.Path(TEX_Wrangle.__file__).resolve().parent.parent)
    try:
        m = _measure("import TEX_Wrangle", custom_nodes)
    except Exception as e:
        r.fail("bare `import TEX_Wrangle`", f"{type(e).__name__}: {e}")
        return
    if m["tex"] <= _BARE_TOUCH_TEX_MODULES_MAX and m["torch"] == _BARE_TOUCH_TORCH_MODULES:
        r.ok(f"bare `import TEX_Wrangle`: {m['tex']} TEX module(s), {m['torch']} torch "
             f"module(s) (PORT-6 floor: <= {_BARE_TOUCH_TEX_MODULES_MAX} / "
             f"== {_BARE_TOUCH_TORCH_MODULES})")
    else:
        r.fail("bare `import TEX_Wrangle` cold-import ratchet",
               f"got tex={m['tex']} (max {_BARE_TOUCH_TEX_MODULES_MAX}), "
               f"torch={m['torch']} (want {_BARE_TOUCH_TORCH_MODULES}) — an eager import "
               f"crept back into the package root")


def test_hostaudit1_tex_engine_import_module_count_ratchet(r):
    """`from TEX_Wrangle import tex_engine` is torch-heavy by necessity (it IS the tensor
    engine) — this pins the TEX-module side only, as a ratchet that may move down."""
    import TEX_Wrangle
    custom_nodes = str(pathlib.Path(TEX_Wrangle.__file__).resolve().parent.parent)
    try:
        m = _measure("from TEX_Wrangle import tex_engine", custom_nodes)
    except Exception as e:
        r.fail("`from TEX_Wrangle import tex_engine`", f"{type(e).__name__}: {e}")
        return
    if m["tex"] <= _TEX_ENGINE_TEX_MODULES_MAX:
        r.ok(f"`from TEX_Wrangle import tex_engine`: {m['tex']} TEX modules "
             f"(ratchet max {_TEX_ENGINE_TEX_MODULES_MAX}), {m['torch']} torch modules "
             f"(informational — torch is load-bearing for the engine itself)")
    else:
        r.fail("tex_engine import module-count ratchet",
               f"got {m['tex']} TEX modules, ratchet max is {_TEX_ENGINE_TEX_MODULES_MAX} — "
               f"an eager import grew the engine's cold-import closure; if intentional, "
               f"lower the ratchet in this test and say why in the commit")
