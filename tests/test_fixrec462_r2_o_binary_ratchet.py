"""FIX-REC (v0.46.2 Phase C) — R2 [altitude]: one binary-safe low-level open helper for state,
ratcheted so a second raw `os.open(` can't recur unnoticed.

RESTORE-462's Windows text-mode fix (`564f674`) landed as a local flag
(`os.O_RDONLY | getattr(os, "O_BINARY", 0)`) at the one call site that needed it
(`tex_recovery._probe_key`), with no routed-through helper — unlike `atomic_write`
(fsync-before-rename) and `bounded_mkstemp` (the retry bound), which this module already
centralizes for exactly this "one place, so the bug class can't recur" reason
(R4-altitude finding #2). FIX-REC moved the call into `tex_recovery._open_state_ro_binary`,
the tree's one low-level binary-mode reader of on-disk state.

This is the ratchet: an AST scan of every tracked `.py` file (product code AND tests — the
same reach the B3 audit used) for real `os.open(` CALL nodes (never a docstring or comment
that merely mentions the text) must find EXACTLY one, and it must be inside
`tex_recovery.py`'s `_open_state_ro_binary`. A second raw `os.open(` added anywhere else —
this module or a sibling one — reds with its `file:line` instead of waiting to be noticed on
the one platform (Windows) the bug can manifest on.
"""
import ast
import os

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # TEX_Wrangle/
_SKIP_DIRS = ("__pycache__", "editor_build", "node_modules", ".git")
_HELPER_MODULE = "tex_recovery.py"
_HELPER_FUNC = "_open_state_ro_binary"


def _iter_py_files():
    for root, dirs, files in os.walk(_PKG):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for fn in files:
            if fn.endswith(".py"):
                yield os.path.join(root, fn)


def _os_open_calls(path):
    """`(enclosing_function_name_or_None, lineno)` for every real `os.open(...)` CALL node in
    one file's AST. AST-based, not textual, so a docstring/comment that merely quotes
    `os.open(` (this file's own module docstring, `tex_recovery._open_state_ro_binary`'s
    docstring) is never mistaken for a call."""
    try:
        with open(path, encoding="utf-8") as f:
            src = f.read()
        tree = ast.parse(src, filename=path)
    except (OSError, UnicodeDecodeError, SyntaxError):
        return []
    hits = []

    class _Visitor(ast.NodeVisitor):
        def __init__(self):
            self.func_stack = []

        def visit_FunctionDef(self, node):
            self.func_stack.append(node.name)
            self.generic_visit(node)
            self.func_stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Call(self, node):
            f = node.func
            if (isinstance(f, ast.Attribute) and f.attr == "open"
                    and isinstance(f.value, ast.Name) and f.value.id == "os"):
                hits.append((self.func_stack[-1] if self.func_stack else None, node.lineno))
            self.generic_visit(node)

    _Visitor().visit(tree)
    return hits


def test_r2_only_one_raw_os_open_in_the_tracked_tree(r):
    print("\n--- FIX-REC R2: no raw os.open( outside the one binary-safe helper ---")
    offenders = []
    helper_hits = 0
    for path in _iter_py_files():
        rel = os.path.relpath(path, _PKG).replace(os.sep, "/")
        for func_name, lineno in _os_open_calls(path):
            if rel.endswith(_HELPER_MODULE) and func_name == _HELPER_FUNC:
                helper_hits += 1
                continue
            offenders.append(f"{rel}:{lineno} (in {func_name or '<module scope>'})")
    if offenders:
        r.fail("FIX-REC R2 ratchet",
               "raw os.open( found outside the binary-safe helper:\n  " + "\n  ".join(offenders))
    elif helper_hits != 1:
        r.fail("FIX-REC R2 ratchet",
               f"expected exactly 1 os.open( call inside {_HELPER_MODULE}::{_HELPER_FUNC}, "
               f"found {helper_hits}")
    else:
        r.ok(f"the tracked tree's only os.open( call is inside {_HELPER_MODULE}::{_HELPER_FUNC}")


def test_r2_helper_ors_in_o_binary(r):
    """Behavioural half: the helper itself still carries the `O_BINARY` bit (inert 0 on POSIX,
    load-bearing on Windows) — the ratchet above only proves there is exactly one call site,
    not that the call site is still correct."""
    print("\n--- FIX-REC R2: _open_state_ro_binary still ORs in O_BINARY ---")
    from TEX_Wrangle import tex_recovery as R
    captured = {}
    real_open = os.open

    def spy(path, flags):
        captured["flags"] = flags
        raise FileNotFoundError(2, "No such file or directory")

    os.open = spy
    try:
        try:
            R._open_state_ro_binary("does-not-exist")
        except FileNotFoundError:
            pass
    finally:
        os.open = real_open
    want_bit = getattr(os, "O_BINARY", 0)
    if "flags" not in captured:
        r.fail("FIX-REC R2 helper", "os.open was never called")
    elif captured["flags"] & os.O_RDONLY != os.O_RDONLY:
        r.fail("FIX-REC R2 helper", f"O_RDONLY not set: flags={captured['flags']!r}")
    elif captured["flags"] & want_bit != want_bit:
        r.fail("FIX-REC R2 helper", f"O_BINARY not set: flags={captured['flags']!r}")
    else:
        r.ok("_open_state_ro_binary opens with O_RDONLY | O_BINARY")
