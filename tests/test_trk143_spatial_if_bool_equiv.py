"""TRK-143 — PARITY-46: the interpreter's `_exec_spatial_if` "is this pixel on" formula and
codegen's `0.23` spatial-if emitter now compute the SAME answer on every condition, integer
and NaN included, because they call the same shared function.

THE FIX (PARITY-46, on top of the row's original finding). `masked_flow.cond_mask` already
existed — TRK-152 built it for the language-0.25 masked `if`, and its own docstring already
named this row's exact hazard as the reason for building one shared helper. What PARITY-46
does is make the UNMASKED `0.23` path on BOTH tiers route through that same helper instead of
each still spelling its own formula:
  * interpreter (`tex_runtime/interpreter_control_flow.py::Interpreter._exec_spatial_if`):
      `cond_bool = _masked_flow_mod.cond_mask(cond)`
  * codegen (`tex_runtime/codegen.py::_CodeGen._emit_spatial_if_else`, the `0.23` path):
      emits `{cond_bool} = _MF.cond_mask({cond_var})` — the identical call the `0.25` masked
      emitter already makes (`codegen_masked.py`'s `_emit(f"{cm} = _MF.cond_mask({cond_var})")`).

`cond_mask` itself is `(cond > 0.5) if cond.is_floating_point() else cond.bool()` (plus a
`torch.bool`-dtype passthrough) — the INTERPRETER's original spelling, adopted deliberately
(see `cond_mask`'s own docstring, and the TRK-152 row's ruling this row's tracker entry
already cites): the spatial `if` uses one mask both to gate the branch and to select the
merge, and the interpreter is the oracle, so changing ITS reading would move a `0.23`
program's answer, which invariant 7 forbids. Codegen's old unconditional `> 0.5` is what
moved instead.

For a FLOATING condition (including NaN) this changes nothing: `cond_mask`'s floating branch
is the literal `cond > 0.5`, the same expression both tiers already used, so NaN still reads
OFF on both — verified below, not assumed.

For a NON-floating condition this is the actual fix: before PARITY-46, `.bool()` (a nonzero
test) and `> 0.5` (a numeric-threshold test) disagreed for a negative nonzero integer
(`.bool()` → True, `> 0.5` → False) — `test_trk143_raw_formula_disagreement_...` used to prove
that disagreement at the raw-tensor level. It now proves the opposite: with M5-INT
deliberately disarmed on BOTH tiers' raw invocation, they still agree, because both paths
reach the exact same `cond_mask` call regardless of M5-INT. The fix closes the hazard at its
root rather than only behind the M5-INT guard.

PIXEL-NEUTRALITY ON EVERY REAL INPUT PATH (M5-INT). Enumerating every path an integer
condition can reach `_exec_spatial_if`/`_emit_spatial_if_else` through:
  1. An integer-dtype IMAGE-LIKE (dim>=3) BINDING, fed straight into the condition
     expression. Both tiers' production ingestion — `Interpreter.execute`'s own binding
     loop, and `tex_runtime.compiled._contiguous_bindings` (the call `compiled.py`'s real
     dispatch makes before a cook) — call `tex_marshalling.to_fp32_if_int_image` (M5-INT)
     on exactly this shape before a program ever runs, so the condition is already float by
     the time either formula runs. `test_trk143_integer_binding_never_reaches_spatial_if_
     unguarded` drives both tiers through this real ingestion and confirms it.
  2. A LOCAL variable derived from integer arithmetic on a binding (e.g. `int k = int(@A.r *
     10.0) - 5;`), never itself a "binding" M5-INT would see. This is exactly the shape the
     tracker row's own probe used (`float k = @A.r; if (k) {...}` — `k` carries through
     whatever dtype the expression produced, unguarded by M5-INT). Before PARITY-46 this WAS
     the live gap: M5-INT only ever normalises `bindings`, never an intermediate local. After
     PARITY-46 it no longer matters whether M5-INT ran — `cond_mask` is now the ONLY formula
     either tier calls, so an int64 local reaches the identical function on both sides and
     both sides return the identical tensor. Proven directly by
     `test_trk143_raw_formula_disagreement_is_now_agreement`, which disarms M5-INT and drives
     an unguarded raw int64 condition through both tiers' bare invocation.
  3. A CONSTANT integer literal folded to a condition (`if (-1) {...}`). The optimizer/
     type-checker promote every numeric literal to a float TEX value before it reaches a
     runtime condition (TEX has no integer literal type distinct from `int`-tagged floats at
     the tensor level — `int` is a per-pixel float tensor with an integer-valued content, not
     a distinct dtype), so this path is the same float-producing shape as case 1, not a
     genuinely non-floating tensor; not a separate case.
Since case 3 collapses into the float case and cases 1 and 2 both now agree unconditionally
(case 1 already agreed before this fix, behind the M5-INT guard; case 2 is what this fix
actually closes), every reachable path agrees after PARITY-46 — not merely the guarded one.

PORTABILITY: CPU only, no ComfyUI, no CUDA, no compiler, no numpy, no timing.
"""
import torch

from helpers import *
from TEX_Wrangle.tex_cache import parse_and_split
from TEX_Wrangle.tex_runtime.codegen import _invoke_cg
from TEX_Wrangle.tex_runtime.compiled import _contiguous_bindings
import TEX_Wrangle.tex_marshalling as _marshalling

#: `k` is a plain FLOAT local, not a binding — so it is never subject to M5-INT itself;
#: it just carries through to the if whatever dtype `@A.r` handed it, unmodified, the
#: same as the tracker's own probe (`int k = (u < 0.5) ? -1 : 0; if (k) {...}`).
_CODE = ("float k = @A.r;\n"
         "if (k) { @OUT = vec4(1.0, 1.0, 1.0, 1.0); }\n"
         "else { @OUT = vec4(0.0, 0.0, 0.0, 0.0); }\n")
_BT = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}
#: Not in `helpers.__all__` (HOOK-4 pins that set), so bound locally rather than via
#: `from helpers import *`.
_STDLIB_FNS = TEXStdlib.get_functions()
_CPU_DEVICE = torch.device("cpu")


def _compile():
    prog = parse_and_split(_CODE, _BT)
    type_map = TypeChecker(binding_types=_BT, source=_CODE).check(prog)
    cg_fn = try_compile(prog, type_map)
    return prog, type_map, cg_fn


def _run_interp_raw(prog, type_map, a_tensor):
    """Interpreter, through its real `execute()` entry — M5-INT stays ARMED."""
    interp = Interpreter()
    out = interp.execute(prog, {"A": a_tensor.clone()}, type_map, device="cpu",
                          output_names=["OUT"])
    return out["OUT"][0, 0, :, 0].tolist()   # the .r channel, one value per pixel


def _run_codegen_bypassed(prog, cg_fn, a_tensor):
    """Codegen, via `_invoke_cg` directly — the SAME call `run_both` uses, which never
    goes through `_contiguous_bindings` (that call lives only in `compiled.py`'s own
    dispatch, not in the raw codegen invocation). M5-INT is therefore never applied on
    this path regardless of its own armed/disarmed state — exactly what lets this probe
    reach the raw emitted formula with a genuinely non-floating tensor."""
    env: dict = {}
    bindings = {"A": a_tensor.clone()}
    _invoke_cg(cg_fn, env, bindings, _STDLIB_FNS, _CPU_DEVICE, (1, 1, 3), program=prog)
    return bindings["OUT"][0, 0, :, 0].tolist()


def _run_codegen_guarded(prog, cg_fn, a_tensor):
    """Codegen, with the SAME M5-INT normalisation `compiled.py`'s own dispatch applies
    before a real cook — the guard every production call site actually gets."""
    env: dict = {}
    bindings = _contiguous_bindings({"A": a_tensor.clone()}, _CPU_DEVICE)
    _invoke_cg(cg_fn, env, bindings, _STDLIB_FNS, _CPU_DEVICE, (1, 1, 3), program=prog)
    return bindings["OUT"][0, 0, :, 0].tolist()


def test_trk143_nan_condition_bool_equiv(r: SubTestResult):
    """PERMANENT GUARD: on a NaN condition both formulas ARE the same expression
    (`cond > 0.5`, which is False for NaN under IEEE 754) — pin the agreement so a
    future change to either tier's NaN handling (e.g. `nan_to_num`, a `.bool()` branch
    added for floats) cannot silently diverge them."""
    print("\n--- TRK-143: NaN condition — interpreter and codegen agree ---")
    try:
        prog, type_map, cg_fn = _compile()
        assert cg_fn is not None, "premise: this program must reach codegen"
        a = torch.zeros(1, 1, 3, 3, dtype=torch.float32)
        a[0, 0, 0, 0] = float("nan")
        a[0, 0, 1, 0] = 0.0
        a[0, 0, 2, 0] = 1.0
        interp_vals = _run_interp_raw(prog, type_map, a)
        cg_vals = _run_codegen_bypassed(prog, cg_fn, a)
        assert interp_vals == cg_vals == [0.0, 0.0, 1.0], (
            f"interp={interp_vals} codegen={cg_vals}, expected [0.0, 0.0, 1.0] on both "
            f"(NaN -> off, 0.0 -> off, 1.0 -> on)")
        r.ok(f"[nan, 0.0, 1.0] -> interp {interp_vals} == codegen {cg_vals}")
    except Exception as e:
        r.fail("TRK-143 NaN bool equiv", f"{type(e).__name__}: {e}")


def test_trk143_integer_binding_never_reaches_spatial_if_unguarded(r: SubTestResult):
    """PERMANENT GUARD: through each tier's REAL production ingestion (M5-INT armed —
    `Interpreter.execute`'s own binding loop; `_contiguous_bindings` on the codegen
    side, the same call `compiled.py`'s dispatch makes before a real cook), a
    non-floating image-like binding is cast to fp32 before either formula runs, so the
    two tiers agree in every reachable case — including the negative-nonzero-integer
    input that, before PARITY-46, the raw formulas (below) did NOT agree on unguarded.
    Since PARITY-46 the raw formulas agree too (one shared `cond_mask` call), so this
    guard is now belt-and-suspenders rather than the only thing standing between the
    hazard and a live program — kept anyway, because M5-INT casting the binding before
    either formula runs is still the real production path and worth pinning on its own."""
    print("\n--- TRK-143: integer binding, real ingestion (M5-INT armed) — still agree ---")
    try:
        prog, type_map, cg_fn = _compile()
        assert cg_fn is not None, "premise: this program must reach codegen"
        a = torch.zeros(1, 1, 3, 3, dtype=torch.int64)
        a[0, 0, 0, 0] = -1   # the exact value the raw formulas disagree on
        a[0, 0, 1, 0] = 0
        a[0, 0, 2, 0] = 2
        interp_vals = _run_interp_raw(prog, type_map, a)      # M5-INT armed here too
        cg_vals = _run_codegen_guarded(prog, cg_fn, a)
        assert interp_vals == cg_vals, (
            f"interp={interp_vals} codegen={cg_vals} — M5-INT should make every "
            f"reachable integer-binding case agree")
        r.ok(f"[-1, 0, 2] (int64), guarded ingestion -> interp {interp_vals} == "
             f"codegen {cg_vals} (both cast to fp32 before the if)")
    except Exception as e:
        r.fail("TRK-143 integer bool equiv (guarded)", f"{type(e).__name__}: {e}")


def test_trk143_raw_formula_disagreement_is_now_agreement(r: SubTestResult):
    """PERMANENT GUARD (was the opposite before PARITY-46): with M5-INT deliberately
    disarmed on BOTH tiers' raw invocation — the shape that used to prove the two
    formulas genuinely disagreed for a negative nonzero integer (`.bool()` is a nonzero
    test; the old codegen `> 0.5` is a numeric threshold) — they now AGREE, because both
    tiers reach the identical `masked_flow.cond_mask` call regardless of M5-INT.
    `-1`: `cond_mask` reads it ON on both tiers (`.bool()` semantics, the interpreter's
    original spelling, kept); `0`: OFF on both; `2`: ON on both. This asserts EQUALITY
    where the row used to pin a documented disagreement — the fix closes the hazard at
    its root (one shared formula), not only behind the M5-INT guard the previous test
    already covers. If this ever reds, the shared-helper invariant itself broke — one
    of the two call sites started spelling the formula itself again."""
    print("\n--- TRK-143: raw formulas, M5-INT disarmed — now agree (PARITY-46) ---")
    saved = _marshalling.to_fp32_if_int_image
    _marshalling.to_fp32_if_int_image = lambda t, device=None: t
    try:
        prog, type_map, cg_fn = _compile()
        assert cg_fn is not None, "premise: this program must reach codegen"
        a = torch.zeros(1, 1, 3, 3, dtype=torch.int64)
        a[0, 0, 0, 0] = -1
        a[0, 0, 1, 0] = 0
        a[0, 0, 2, 0] = 2
        interp_vals = _run_interp_raw(prog, type_map, a)          # M5-INT disarmed
        cg_vals = _run_codegen_bypassed(prog, cg_fn, a)           # never armed on this path
        assert interp_vals == cg_vals == [1.0, 0.0, 1.0], (
            f"interp={interp_vals} codegen={cg_vals}, expected [1.0, 0.0, 1.0] on both "
            f"(-1 -> on, 0 -> off, 2 -> on, per cond_mask's nonzero-test reading)")
        r.ok(f"[-1, 0, 2] (int64), M5-INT disarmed -> interp {interp_vals} == codegen "
             f"{cg_vals} (PARITY-46: one shared cond_mask formula, not two)")
    except Exception as e:
        r.fail("TRK-143 raw formula agreement", f"{type(e).__name__}: {e}")
    finally:
        _marshalling.to_fp32_if_int_image = saved
