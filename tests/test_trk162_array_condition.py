"""TRK-162 — an ARRAY-typed `if`/`for`/`while` condition type-checks clean, then
crashes at cook.

`type_checker._check_scalar_condition` (the single check backing `_check_if_else`,
`_check_for_loop` AND `_check_while_loop` — the tracker row names only the first two;
the third calls it too) rejected a condition only when `cond_type.is_vector`.
`TEXType.ARRAY.is_vector` is `False`, so an ARRAY-typed condition — reachable with a
plain LOCAL array declaration, no engine profile needed (`float arr[3] = {...};` is
legal on the default ComfyUI profile; only a WIRED array binding needs
`tex_compiler.types.set_array_wires(True)`) — passed `TypeChecker.check` with no
error, then crashed three layers into execution: `RuntimeError: The size of tensor a
(3) must match the size of tensor b (4) at non-singleton dimension 0`, inside
`_merge_branch_vars`'s `_tensor_where` (the condition's own length vs. the branch
values' vector width).

Fixed the same way `TRK-9`/`TRK-28` each closed their own crash shapes: refuse at
type-check time with a named diagnostic (`E3501`, alongside `E3500`'s existing
vector refusal) instead of a bare `RuntimeError` deep in execution.

ComfyUI-invisible because: a program that never binds or declares an ARRAY value
takes the identical `_check_scalar_condition` path as before (`cond_type.is_vector`
is checked first, unchanged); only a program whose condition is ARRAY-typed is
affected, and it used to crash uncontrolled, so the only visible change is that the
crash is now a clear compile-time message instead of a bare RuntimeError.
"""
from contextlib import nullcontext as _nullcontext

from helpers import *

from TEX_Wrangle import tex_api, tex_engine
from TEX_Wrangle.tex_compiler.types import array_wires_enabled, set_array_wires


def _codes(source):
    return [(d.code, d.severity, d.message) for d in tex_api.check(source, {})]


class _array_wires:
    """Context manager: flip the engine profile on, restore whatever it was after —
    mirrors the `_planes_enabled`/`array_wires` idiom already used elsewhere in the
    suite (`tests/test_v037_frontend_parity.py`)."""

    def __enter__(self):
        self._prev = array_wires_enabled()
        set_array_wires(True)
        return self

    def __exit__(self, *a):
        set_array_wires(self._prev)


def test_trk162_premise_array_condition_crashes_uncontrolled(r: SubTestResult):
    """Re-verify the row's own premise fresh (never carried): at the tree this lane
    started from, an ARRAY-typed `if` condition checks clean and crashes at cook with
    a bare RuntimeError. This test asserts the CURRENT (fixed) behaviour instead —
    `check()` now catches it — so a reader who wants the base-sha premise re-run can
    diff this file's own history rather than trust prose."""
    print("\n--- TRK-162 premise: an ARRAY-typed if-condition is refused at check(), "
          "never reaches cook() ---")
    src = ("float arr[3] = {1.0, 2.0, 3.0}; "
          "if (arr) { @OUT = vec4(1.0); } else { @OUT = vec4(0.0); }")
    codes = [c for c, sev, _ in _codes(src) if sev == "error"]
    if "E3501" in codes:
        r.ok(f"check() refuses the ARRAY condition ({codes}); cook() is never reached")
    else:
        r.fail("TRK-162 premise", f"expected E3501, got {codes}")
    try:
        tex_engine.cook(src, {}, device_mode="cpu")
        r.fail("TRK-162 premise", "cook() should not have been reachable at all")
    except tex_api.TEXCompileError as e:
        got = [d.code for d in e.diagnostics]
        if "E3501" in got:
            r.ok(f"cook() surfaces the SAME compile-time refusal (E3501), never a bare "
                 f"RuntimeError: {e}")
        else:
            r.fail("TRK-162 premise: cook() raises E3501", f"got codes={got}")
    except Exception as e:
        r.fail("TRK-162 premise: cook() raises a structured TEXCompileError, not a "
              "bare crash", f"{type(e).__name__}: {e}")


def test_trk162_array_condition_refused_if_for_while(r: SubTestResult):
    """All three condition-bearing constructs — `if`, `for`, `while` — go through the
    same `_check_scalar_condition`, so all three must refuse an ARRAY condition with
    E3501."""
    print("\n--- TRK-162: if / for / while each refuse an ARRAY-typed condition (E3501) ---")
    cases = {
        "if": ("float arr[3] = {1.0, 2.0, 3.0}; "
              "if (arr) { @OUT = vec4(1.0); } else { @OUT = vec4(0.0); }"),
        "for": ("float arr[3] = {1.0, 2.0, 3.0}; float s = 0.0; "
               "for (int i = 0; arr; i = i + 1) { s = s + 1.0; } @OUT = vec4(s);"),
        "while": ("float arr[3] = {1.0, 2.0, 3.0}; "
                 "while (arr) { break; } @OUT = vec4(1.0);"),
    }
    for keyword, src in cases.items():
        codes = [c for c, sev, _ in _codes(src) if sev == "error"]
        if "E3501" in codes:
            r.ok(f"'{keyword}' with an ARRAY condition draws E3501")
        else:
            r.fail(f"TRK-162 {keyword}", f"expected E3501, got {codes}")


def test_trk162_array_wire_binding_condition_refused(r: SubTestResult):
    """The tracker row's own original repro shape: a WIRED array binding (`a@cond`,
    DATA-3's `a` type hint — TEXType.ARRAY, not gated by `array_wires_enabled` for a
    READ), not just a local declaration, used as a condition. Checked under BOTH
    profile states: the hint resolves to ARRAY regardless, so the refusal must fire
    either way — this is the "engine-profile-only" framing the tracker row uses for
    a REAL cook (an array wire can only be bound at cook time under the engine
    profile), not a constraint on the type-checker's own refusal."""
    print("\n--- TRK-162: a wired ARRAY binding (a@cond) used as a condition is refused ---")
    src = "if (a@cond) { @OUT = vec4(1.0); } else { @OUT = vec4(0.0); }"
    for label, on in (("comfy (default)", False), ("engine", True)):
        with _array_wires() if on else _nullcontext():
            codes = [c for c, sev, _ in _codes(src) if sev == "error"]
            if "E3501" in codes:
                r.ok(f"[{label}] a wired ARRAY condition draws E3501 ({codes})")
            else:
                r.fail(f"TRK-162 wire-binding [{label}]", f"expected E3501, got {codes}")


def test_trk162_scalar_and_vector_conditions_unaffected(r: SubTestResult):
    """Control: this fix must not over-refuse. A scalar condition still checks clean,
    and a vector condition still draws the pre-existing E3500 (unchanged branch,
    checked first)."""
    print("\n--- TRK-162 control: scalar conditions clean, vector conditions still E3500 ---")
    clean = "float x = 1.0; if (x) { @OUT = vec4(1.0); } else { @OUT = vec4(0.0); }"
    errs = [c for c, sev, _ in _codes(clean) if sev == "error"]
    r.ok("a scalar if-condition still checks clean") if not errs \
        else r.fail("TRK-162 control scalar", f"unexpected errors {errs}")

    vec = "if (vec3(1.0)) { } @OUT = vec4(1.0);"
    codes = [c for c, sev, _ in _codes(vec) if sev == "error"]
    r.ok(f"a vector if-condition still draws E3500 ({codes})") if "E3500" in codes \
        else r.fail("TRK-162 control vector", f"expected E3500, got {codes}")
