"""TRK-143 — pin the interpreter's `_exec_spatial_if` "is this pixel on" formula and
codegen's `0.23` spatial-if emitter to the same answer, on integer and NaN conditions.

THE TWO FORMULAS, quoted from the tracker row. The interpreter one was quoted from
`tex_runtime/interpreter.py` when the row was written; SPLIT-I (`main`, after this row)
moved if/for/while execution out to `tex_runtime/interpreter_control_flow.py` as a
byte-identical, mechanical move — `Interpreter` still has the method through the mixin,
and this file drives it only through `Interpreter.execute()`, never by importing the
symbol directly, so the move needed no code change here, only this citation:
  * interpreter (`tex_runtime/interpreter_control_flow.py::Interpreter._exec_spatial_if`):
      `cond_bool = (cond > 0.5) if cond.is_floating_point() else cond.bool()`
  * codegen (`tex_runtime/codegen.py::_CodeGen._emit_spatial_if_else`, the `0.23` path):
      `{cond_bool} = ({cond_var} > 0.5)` — unconditionally, regardless of dtype.

For a FLOATING condition (including NaN) both formulas are the literal same expression
(`cond > 0.5`), so they can never disagree there — verified below, not assumed.

For a NON-floating condition they are NOT the same formula: `.bool()` is a nonzero test,
`> 0.5` is a numeric-threshold test, and they disagree for a negative nonzero integer
(`.bool()` → True, `> 0.5` → False). `test_trk143_raw_formula_disagreement_...` below
proves this is a REAL disagreement at the raw-tensor level — a genuine invariant-2
hazard — filed as a finding rather than fixed here (fixing the emitter moves emitted
bytes and needs its own lane, per the ask).

WHY IT IS LATENT, NOT LIVE, and pinned as a permanent guard here: a non-floating
IMAGE-LIKE (dim>=3) binding never reaches either formula through any real entry point.
Both tiers' own production ingestion — `Interpreter.execute`'s binding loop and
`tex_runtime.compiled._contiguous_bindings` — call the ONE shared
`tex_marshalling.to_fp32_if_int_image` (M5-INT) before a program ever runs, which casts
exactly this shape to fp32. `test_trk143_integer_binding_never_reaches_spatial_if_...`
below drives BOTH tiers through their real ingestion (the cast left ARMED) and confirms
they agree — because by the time either formula runs, the condition is already float.
That guard is what this row closes with evidence: if M5-INT is ever weakened or
bypassed on one tier only, this test goes red.

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
    input that the raw formulas (next test) do NOT agree on."""
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


def test_trk143_raw_formula_disagreement_is_a_real_but_latent_invariant2_hazard(r: SubTestResult):
    """NOT a permanent guard — the opposite: proves, with M5-INT deliberately disarmed
    on BOTH tiers' raw invocation, that the two formulas themselves genuinely disagree
    for a negative nonzero integer (`.bool()` is a nonzero test; `> 0.5` is a numeric
    threshold). `-1`: `.bool()` -> True (interpreter selects the THEN branch); `-1 >
    0.5` -> False (codegen selects the ELSE branch). This is the invariant-2 hazard
    TRK-143's row exists to check for — reported separately, per the ask, rather than
    fixed here (changing codegen's emitter moves emitted bytes and needs its own lane).
    Kept as a pinned, monitored fact rather than an assumption: if this ever changes in
    either direction
    (the emitter changes, or the interpreter's formula changes), this test will notice
    and the row's status needs re-deciding, not silently updating."""
    print("\n--- TRK-143: raw formulas, M5-INT disarmed — a real, latent disagreement ---")
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
        assert interp_vals == [1.0, 0.0, 1.0], (
            f"expected the interpreter's `.bool()` formula to read -1 as ON: {interp_vals}")
        assert cg_vals == [0.0, 0.0, 1.0], (
            f"expected codegen's `> 0.5` formula to read -1 as OFF: {cg_vals}")
        assert interp_vals != cg_vals, (
            "the two formulas stopped disagreeing on -1 — TRK-143's row needs "
            "re-deciding (this would be GOOD news: re-check with M5-INT disarmed and "
            "update the tracker row rather than silently accepting a green here)")
        r.ok(f"[-1, 0, 2] (int64), M5-INT disarmed -> interp {interp_vals} != codegen "
             f"{cg_vals}; confirmed latent, not fixed (invariant-2 hazard, reported "
             f"separately)")
    except Exception as e:
        r.fail("TRK-143 raw formula disagreement", f"{type(e).__name__}: {e}")
    finally:
        _marshalling.to_fp32_if_int_image = saved
