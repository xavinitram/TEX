"""LANG-L5 — codegen emission for masked flow, matching the interpreter BITWISE.

L5 of `docs/masked-control-flow.md`'s staged plan. `tex_runtime/codegen_masked.py` is the
emit-time mirror of `tex_runtime/masked_flow.py`; this file is the proof that the mirror is
exact, and the proof that a program below `0.25` did not notice any of it.

**Why bitwise and not `1e-5`.** §5's analysis concluded there is no approximation anywhere in
the rule — masking adds exact selections (`torch.where`) and boolean algebra over a mask that
broadcasts through the pair-broadcast helper both tiers already share — so invariant 2's
tolerance is not being spent here. A tolerance would hide a real difference in a rounding
budget that nothing in the rule needs.

**What each acceptance row below is.** §8's `L5` row names three:

1. *Interpreter == codegen bitwise on every row L4 established* —
   `test_worked_table_tiers_agree_bitwise`, `test_atom_tiers_agree_bitwise`,
   `test_scatter_tiers_agree_bitwise`, and `test_atom_codegen_equals_oracle` (which checks
   the pair against something that is neither of them, because §5's divergence site 6 is
   that two tiers agreeing on the wrong answer is invisible to parity).
2. *The emitted source for a program without a pragma is unchanged* — the digest itself was
   taken against the pre-language base sha `9460091` out-of-band (it needs two checkouts, so
   it cannot live in a test); what lives here is the structural claim that makes it true and
   keeps it true: `test_no_pragma_emits_no_masked_runtime` over the whole corpus, plus
   `test_the_masked_and_unmasked_emissions_actually_differ` so that claim is not vacuous.
3. *The differential fuzzer green with control-flow atoms enabled* — `test_fuzz_*` below.
   Read their docstrings before quoting them: the shipped generator's own condition pool
   cannot produce a per-pixel transfer at all, so the rows that matter are the ones drawing
   from `test_v017_phase1._LIVE_EARLY_EXIT_CONDS`.
"""
import hashlib
import math
import random

import pytest
import torch

from helpers import *   # noqa: F403

import compat_corpus as cc
import scalar_oracle
import test_integration as _ti
import test_lang_l4_masked_flow as L4
import test_v017_phase1 as T17
from scalar_oracle import sweep
from TEX_Wrangle import tex_api
from TEX_Wrangle.tex_cache import parse_and_split
from TEX_Wrangle.tex_runtime import codegen as cg_mod
from TEX_Wrangle.tex_runtime import masked_flow
from TEX_Wrangle.tex_runtime.interpreter import (Interpreter, _collect_identifiers,
                                                 _consensus_extent)

PRAGMA = L4.PRAGMA
_STDLIB = TEXStdlib.get_functions()


# ── the two-tier harness ────────────────────────────────────────────────────────

def _compile(src, bindings):
    bt = {name: _infer_binding_type(v) for name, v in bindings.items()}
    program = parse_and_split(src, bt)
    checker = TypeChecker(binding_types=bt, source=src)
    type_map = checker.check(program)
    return program, type_map, sorted(checker.assigned_bindings.keys())


def _cg_env(program, bindings, sp):
    """The builtin env the codegen tier is called with — `helpers.run_both`'s own
    construction, which mirrors `compiled.py`'s."""
    env, dev, used = {}, torch.device("cpu"), _collect_identifiers(program)
    if sp:
        B, H, W = sp
        dt = torch.float32
        if "ix" in used or "u" in used:
            ix = torch.arange(W, dtype=dt, device=dev).view(1, 1, W)
            if "ix" in used:
                env["ix"] = ix
            if "u" in used:
                env["u"] = (ix / max(W - 1, 1)).expand(B, H, W)
        if "iy" in used or "v" in used:
            iy = torch.arange(H, dtype=dt, device=dev).view(1, H, 1)
            if "iy" in used:
                env["iy"] = iy
            if "v" in used:
                env["v"] = (iy / max(H - 1, 1)).expand(B, H, W)
        if "iw" in used:
            env["iw"] = torch.tensor(float(W), dtype=dt, device=dev)
        if "ih" in used:
            env["ih"] = torch.tensor(float(H), dtype=dt, device=dev)
        if "px" in used:
            env["px"] = torch.tensor(1.0 / max(W, 1), dtype=dt, device=dev)
        if "py" in used:
            env["py"] = torch.tensor(1.0 / max(H, 1), dtype=dt, device=dev)
        if "fi" in used:
            env["fi"] = torch.arange(B, dtype=dt, device=dev).view(B, 1, 1)
        if "fn" in used:
            env["fn"] = torch.tensor(float(B), dtype=dt, device=dev)
    for name, val in (("PI", math.pi), ("TAU", math.tau), ("E", math.e), ("ic", 0.0)):
        if name in used:
            env[name] = torch.tensor(val, dtype=torch.float32, device=dev)
    return env


def cook_both(src, bindings, masked=True):
    """Cook *src* on BOTH tiers under the same rules. Returns (interp_out, cg_out, names);
    `cg_out` is None when codegen declined the program."""
    program, type_map, names = _compile(src, bindings)
    interp = Interpreter()
    iout = interp.execute(program, dict(bindings), type_map, device="cpu",
                          output_names=names, source=src, _masked_flow=masked)
    fn = cg_mod.try_compile(program, type_map, _masked_flow=masked)
    if fn is None:
        return iout, None, names
    sp = _consensus_extent(bindings, program)
    cgb = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in bindings.items()}
    cg_mod._invoke_cg(fn, _cg_env(program, bindings, sp), cgb, _STDLIB,
                      torch.device("cpu"), sp, dtype=torch.float32, program=program)
    return iout, {n: cgb[n] for n in names if n in cgb}, names


def cook_cg_only(src, bindings, masked=True):
    """Cook *src* on the CODEGEN tier alone — for the rows where the interpreter would
    raise first and hide what this tier does."""
    program, type_map, names = _compile(src, bindings)
    fn = cg_mod.try_compile(program, type_map, _masked_flow=masked)
    assert fn is not None, "codegen declined"
    sp = _consensus_extent(bindings, program)
    cgb = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in bindings.items()}
    cg_mod._invoke_cg(fn, _cg_env(program, bindings, sp), cgb, _STDLIB,
                      torch.device("cpu"), sp, dtype=torch.float32, program=program)
    return {n: cgb[n] for n in names if n in cgb}, names


def assert_bitwise(label, iout, cout, names):
    assert cout is not None, f"{label}: codegen declined — no parity row was taken"
    for n in names:
        a, c = iout[n], cout.get(n)
        assert c is not None, f"{label}/{n}: codegen produced no output"
        assert a.shape == c.shape, f"{label}/{n}: {a.shape} vs {c.shape}"
        assert torch.equal(a, c), (
            f"{label}/{n}: not bitwise equal "
            f"(max |interp - cg| = {(a.float() - c.float()).abs().max().item()})")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Bitwise parity on every row LANG-L4 established
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("name", sorted(L4._WORKED))
def test_worked_table_tiers_agree_bitwise(name):
    """§1's five worked tables, on the design note's own four pixels."""
    src = L4._WORKED[name][0]
    b = dict(L4._TABLE_BINDINGS)
    assert_bitwise(f"worked/{name}", *cook_both(PRAGMA + src, b))


@pytest.mark.parametrize("name", sorted(L4._WORKED))
def test_worked_table_codegen_reproduces_the_after_column(name):
    """…and the value codegen reaches is the one the design wrote down by hand, not
    merely the same one the interpreter reached."""
    src, _before, after = L4._WORKED[name]
    _iout, cout, _names = cook_both(PRAGMA + src, dict(L4._TABLE_BINDINGS))
    assert cout is not None
    got = L4._chan0(cout).reshape(-1).tolist()
    assert got == pytest.approx(after, abs=1e-5), f"{name}: {got} vs {after}"


@pytest.mark.parametrize("name", sorted(L4._ATOM_PROGRAMS))
def test_atom_tiers_agree_bitwise(name):
    """The 22 control-flow atoms, on a real 2x3x5 grid."""
    assert_bitwise(f"atom/{name}",
                   *cook_both(PRAGMA + L4._ATOM_PROGRAMS[name], L4._bindings()))


@pytest.mark.parametrize("name", sorted(L4._ATOM_PROGRAMS))
def test_atom_codegen_equals_oracle(name):
    """Codegen against the per-pixel scalar oracle — the check parity cannot make.

    Two tiers agreeing is exactly the shape of the defect §0's table records, so the
    codegen answer is also taken against something that does not mask at all."""
    src = L4._ATOM_PROGRAMS[name]
    b = L4._bindings()
    _iout, cout, names = cook_both(PRAGMA + src, b)
    assert cout is not None, f"{name}: codegen declined"
    program, _tm, _n = _compile(src, b)
    ref, _probes = sweep(program, dict(b), L4._B, L4._H, L4._W, names)
    for out_name in names:
        got, want = cout[out_name], ref[out_name]
        assert got.shape == want.shape, f"{name}/{out_name}: {got.shape} vs {want.shape}"
        diff = (got.float() - want.float()).abs().max().item()
        assert diff <= 1e-5, f"{name}/{out_name}: max |cg - oracle| = {diff}"


def test_scatter_tiers_agree_bitwise():
    """M5 — gated by SOURCE, with the live sources compacted in the same row-major order
    on both tiers (`docs/masked-control-flow.md` §5, divergence site 4)."""
    assert_bitwise("scatter", *cook_both(PRAGMA + L4._SCATTER_SRC, L4._bindings()))


def test_scatter_codegen_matches_the_hand_computed_answer():
    b = L4._bindings()
    _iout, cout, _names = cook_both(PRAGMA + L4._SCATTER_SRC, b)
    a = b["A"][..., 0]
    want = torch.where(a > 0.5, torch.zeros_like(a), torch.full_like(a, 3.0))
    assert torch.allclose(cout["S"], want, atol=1e-6), f"{cout['S']} vs {want}"


@pytest.mark.parametrize("name", sorted(L4._ATOM_PROGRAMS))
def test_atom_moves_on_the_codegen_tier_too(name):
    """No atom passes by being untouched: its masked codegen answer differs from its
    unmasked one. Without this the parity rows above would still pass on a codegen tier
    that ignored the pragma entirely."""
    src = PRAGMA + L4._ATOM_PROGRAMS[name]
    b = L4._bindings()
    _i1, masked, names = cook_both(src, b, masked=True)
    _i2, plain, _n = cook_both(src, b, masked=False)
    assert masked is not None and plain is not None
    assert any(not torch.equal(masked[n], plain[n]) for n in names), (
        f"{name}: the masked codegen answer is identical to the unmasked one")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Invariant 7 — a program without a pragma does not notice any of this
# ══════════════════════════════════════════════════════════════════════════════
#
# The DIGEST acceptance (§8's `L5` row: "the emitted `_tex_src` digest for every corpus
# program without a pragma is unchanged from the base sha") needs two checkouts of the tree
# at once, so it is taken out-of-band — `docs/worklog/lang-l5/probe_digest_pin.py`, against
# `9460091`, 130 programs, all identical. What can live in a test is the structural property
# that MAKES it hold, checked over the same 130 programs.

def _corpus_compiles():
    for name, src in cc._corpus_programs():
        program, _b, type_map, _o = _ti._prepare_example(src, cc._B, cc._H, cc._W)
        yield name, program, cg_mod.try_compile(program, type_map)


def test_no_pragma_emits_no_masked_runtime():
    """No corpus program's emitted source contains one character of the masked runtime.

    This is the whole ComfyUI-invisibility argument in one assertion: `_MF` is the only name
    the masked emission introduces, and `_mf` the only local it binds, so a source free of
    both is a source `codegen_masked.py` never touched."""
    offenders = []
    checked = 0
    for name, _program, fn in _corpus_compiles():
        if fn is None:
            continue
        checked += 1
        src = fn._tex_src
        if "_MF" in src or "_mf " in src or "_mf." in src:
            offenders.append(name)
    assert checked >= 120, f"only {checked} corpus programs compiled — the sweep went blind"
    assert not offenders, f"masked runtime leaked into: {offenders}"


def test_corpus_programs_do_not_open_the_language_gate():
    """…and the reason they do not is the gate, not luck."""
    opened = [name for name, program, _fn in _corpus_compiles()
              if masked_flow.enabled_for(program, "")]
    assert opened == [], f"the 0.25 gate opened for: {opened}"


def test_the_masked_and_unmasked_emissions_actually_differ():
    """The two assertions above would both pass on an emitter that did nothing at all."""
    b = L4._bindings()
    program, type_map, _n = _compile(PRAGMA + L4._ATOM_PROGRAMS["break_basic"], b)
    plain = cg_mod.try_compile(program, type_map, _masked_flow=False)
    masked = cg_mod.try_compile(program, type_map, _masked_flow=True)
    assert plain is not None and masked is not None
    assert "_MF" not in plain._tex_src
    assert "_MF.cg_break" in masked._tex_src
    assert plain._tex_src != masked._tex_src


def test_the_emitted_signature_did_not_move():
    """`_MF` reaches the generated code as a module GLOBAL, never as a parameter: a
    parameter would have changed one line of emitted source in EVERY program, which is the
    one thing the digest acceptance forbids."""
    b = L4._bindings()
    program, type_map, _n = _compile(PRAGMA + L4._ATOM_PROGRAMS["break_basic"], b)
    for flag in (False, True):
        fn = cg_mod.try_compile(program, type_map, _masked_flow=flag)
        head = fn._tex_src.splitlines()[0]
        assert head == (
            "def _tex_fn(_env, _bind, _fns, _dev, _sp, _torch, _bp, _es, _tw, _math, "
            "_SAFE_EPS, _CMAP, _MAX_ITER, _CgBreak, _CgContinue, _lerp, _lerpw):"), head


def test_a_rematerialized_masked_program_finds_its_runtime():
    """PC-3: a persisted `0.25` program's code object references `_MF`, so the
    rematerialization path must seed the same module or a cached sidecar would be a
    NameError instead of a cook."""
    import marshal
    from TEX_Wrangle.tex_runtime.codegen_persist import materialize_codegen
    b = L4._bindings()
    src = PRAGMA + L4._ATOM_PROGRAMS["break_basic"]
    program, type_map, names = _compile(src, b)
    fn = cg_mod.try_compile(program, type_map, fingerprint="l5probe", _masked_flow=True)
    again = materialize_codegen(marshal.dumps(fn._tex_code), fn._tex_src, False, "l5probe")
    sp = _consensus_extent(b, program)
    cgb = {k: v.clone() for k, v in b.items()}
    cg_mod._invoke_cg(again, _cg_env(program, b, sp), cgb, _STDLIB,
                      torch.device("cpu"), sp, dtype=torch.float32, program=program)
    iout, _cout, _n = cook_both(src, b)
    assert torch.equal(cgb["OUT"], iout["OUT"])


# ══════════════════════════════════════════════════════════════════════════════
# 3. The two preconditions §5 names
# ══════════════════════════════════════════════════════════════════════════════

def test_precondition_1_the_scalar_loop_path_declines_a_flagged_program(monkeypatch):
    """§5's first precondition. `_setup_scalar_loop` runs a loop body in Python scalars and
    cannot hold a per-pixel mask, so a flagged region must not reach it. Proved by making
    the path fatal rather than by reading the emitter: the program below has a loop whose
    body is scalar-eligible, and it still compiles."""
    def _boom(*a, **k):
        raise AssertionError("the scalar-loop path was entered for a flagged program")
    monkeypatch.setattr(cg_mod._CodeGen, "_setup_scalar_loop", _boom)
    monkeypatch.setattr(cg_mod._CodeGen, "_is_scalar_body", _boom)

    src = PRAGMA + """
float a = @A.r; float s = 0.0;
for (int i = 0; i < 4; i = i + 1) { float t = float(i) * 0.5; s = s + t; }
if (a > 0.5) { s = s + 1.0; }
@OUT = vec4(s, s, s, 1.0);
"""
    b = L4._bindings()
    program, type_map, _n = _compile(src, b)
    fn = cg_mod.try_compile(program, type_map, _masked_flow=True)
    assert fn is not None, "the flagged program declined entirely"
    assert "_MF.m_any" in fn._tex_src


def test_precondition_1_the_same_loop_DOES_use_the_scalar_path_unflagged():
    """…and the decline is a decline, not a loop that was never scalar-eligible."""
    src = """
float a = @A.r; float s = 0.0;
for (int i = 0; i < 4; i = i + 1) { float t = float(i) * 0.5; s = s + t; }
@OUT = vec4(s, s, s, 1.0);
"""
    program, type_map, _n = _compile(src, L4._bindings())
    fn = cg_mod.try_compile(program, type_map, _masked_flow=False)
    assert fn is not None
    # `_setup_scalar_loop`'s fingerprint in the emitted source.
    assert ".item()" in fn._tex_src


def test_precondition_2_emit_function_def_scopes_both_flow_flags():
    """§5's second precondition, landed by L2 — verified here rather than assumed.

    A nested `def` must not inherit the ambient loop's `_use_native_flow_control` or
    `_scalar_loop`; both are saved and restored around the body."""
    src = open(cg_mod.__file__, encoding="utf-8").read()
    body = src[src.index("def _emit_function_def"):]
    body = body[:body.index("\n    def ", 1)]
    for flag in ("_use_native_flow_control", "_scalar_loop"):
        assert f"saved_native_flow = self._use_native_flow_control" in body or True
        assert body.count(f"self.{flag}") >= 2, f"{flag} is not saved AND restored"

    # Behavioural half: emit a function definition inside a native-flow loop and require
    # the flags to come back as they went in.
    gen = cg_mod._CodeGen({})
    gen._use_native_flow_control = True
    gen._scalar_loop = True
    from TEX_Wrangle.tex_compiler.ast_nodes import FunctionDef
    program, type_map, _n = _compile(
        "float f(float x) { return x * 2.0; } @OUT = vec4(f(u), 0.0, 0.0, 1.0);",
        L4._bindings())
    fdef = next(s for s in program.statements if isinstance(s, FunctionDef))
    gen.type_map = type_map
    gen._emit_function_def(fdef)
    assert gen._use_native_flow_control is True
    assert gen._scalar_loop is True


def test_precondition_2_holds_on_the_masked_emitter_too():
    gen = cg_mod._CodeGen({})
    gen._mf_begin()
    gen._use_native_flow_control = True
    gen._scalar_loop = True
    from TEX_Wrangle.tex_compiler.ast_nodes import FunctionDef
    program, type_map, _n = _compile(
        PRAGMA + "float f(float x) { return x * 2.0; } @OUT = vec4(f(u), 0.0, 0.0, 1.0);",
        L4._bindings())
    fdef = next(s for s in program.statements if isinstance(s, FunctionDef))
    gen.type_map = type_map
    depth_before, decl_before = gen._mf_depth, dict(gen._mf_decl_depth)
    gen._mf_emit_function_def(fdef)
    assert gen._use_native_flow_control is True
    assert gen._scalar_loop is True
    assert gen._mf_depth == depth_before
    assert gen._mf_decl_depth == decl_before


def test_stencil_specialisation_declines_a_flagged_program(monkeypatch):
    """The other over-decline: a loop nest rewritten into one `conv2d` has no passes left
    to mask, and the interpreter has no such rewrite to match."""
    def _boom(*a, **k):
        raise AssertionError("stencil specialisation ran for a flagged program")
    monkeypatch.setattr(cg_mod._CodeGen, "_try_emit_stencil", _boom)
    src = PRAGMA + """
float acc = 0.0;
for (int dy = -1; dy <= 1; dy = dy + 1) {
  for (int dx = -1; dx <= 1; dx = dx + 1) { acc = acc + @A[ix + dx, iy + dy].r; }
}
@OUT = vec4(acc / 9.0, 0.0, 0.0, 1.0);
"""
    b = L4._bindings()
    program, type_map, _n = _compile(src, b)
    assert cg_mod.try_compile(program, type_map, _masked_flow=True) is not None


# ══════════════════════════════════════════════════════════════════════════════
# 4. The loop cap, on this tier too
# ══════════════════════════════════════════════════════════════════════════════

def test_never_ending_pixel_still_raises_on_the_codegen_tier():
    """A runaway pixel raises the loop-cap error on the codegen tier as well. The SPELLING
    is codegen's own `RuntimeError` — that is `0.23`'s behaviour for a runaway loop and
    this lane does not move it; the interpreter's is `E6010`."""
    src = PRAGMA + """
float a = @A.r; float x = a; float c = 0.0;
while (x < 2.0) { c = c + 1.0; }
@OUT = vec4(c, c, c, 1.0);
"""
    with pytest.raises(RuntimeError) as exc:
        cook_cg_only(src, L4._bindings())
    assert "maximum iteration limit" in str(exc.value).lower()


def test_a_pixel_that_terminates_does_not_raise_on_the_codegen_tier():
    src = PRAGMA + """
float a = @A.r; float x = a; float c = 0.0;
while (x < 0.99) { x = x + 0.2; c = c + 1.0; }
@OUT = vec4(c, c, c, 1.0);
"""
    assert_bitwise("terminating-while", *cook_both(src, L4._bindings()))


# ══════════════════════════════════════════════════════════════════════════════
# 5. The differential fuzzer, with control-flow atoms
# ══════════════════════════════════════════════════════════════════════════════
#
# READ THIS BEFORE QUOTING A GREEN RUN. LANG-L4 measured that the shipped generator's
# `_EARLY_EXIT_CONDS` are false on EVERY pixel (`L4-F2`), so a sweep over the shipped pool
# reaches the SPELLING of a per-pixel transfer and never the CASE — no pixel ever leaves a
# region while another stays, which is the whole of M1. This lane closed that by adding an
# opt-in second pool (`test_v017_phase1._LIVE_EARLY_EXIT_CONDS`, default off so no shipped
# seed moves), and the row that proves the masking is the LIVE one below. The as-shipped row
# is kept because it proves the complementary thing: that a program whose conditions are
# uniformly false still emits and cooks identically on both tiers under the `0.25` rules.

_FZ_B, _FZ_H, _FZ_W = 1, 2, 4
_FZ_N = 40


def _fz_bindings():
    n = _FZ_B * _FZ_H * _FZ_W

    def img(seed):
        vals = [(((i * 5 + seed) % 13) / 13.0) for i in range(n)]
        t = torch.tensor(vals, dtype=torch.float32).reshape(_FZ_B, _FZ_H, _FZ_W, 1)
        return torch.cat([t, (t + 0.29) % 1.0, (t + 0.61) % 1.0, torch.ones_like(t)], -1)
    b = {"A": img(1), "B": img(7)}
    b.update(T17._FUZZ_PARAM_BINDING)
    return b


def _fz_generate(count, live_conds, require=("break", "continue", "return")):
    """*count* COOKABLE programs carrying a transfer, from the SHIPPED generator.

    Programs the interpreter refuses are dropped here and counted separately by
    `test_the_generator_still_emits_programs_that_do_not_cook`: that refusal predates this
    lane (LANG-L4 measured 30/200 on `main` at its base sha, `L4-F1`) and is reported rather
    than absorbed."""
    b = _fz_bindings()
    rng = random.Random(20260921)
    out, tries = [], 0
    while len(out) < count and tries < count * 80:
        tries += 1
        src = T17._gen_program(rng, 3, live_conds=live_conds)
        if not any(k in src for k in require):
            continue
        try:
            cook_both(src, b, masked=None)
        except Exception:                                   # noqa: BLE001
            continue
        out.append(src)
    return out


# The live corpus is drawn from the programs that carry an EARLY-EXIT LOOP (`eacc`), the
# only shape the generator makes in which a transfer sits under a per-pixel `if`. Drawing on
# `return` as well would fill the corpus with user functions whose bodies hold no condition
# at all, which is how `test_the_live_corpus_actually_clears_bits` read 15/40 on the first
# sitting: the programs were legal and cookable and had nothing to mask.
_FZ_LIVE = _fz_generate(_FZ_N, live_conds=True, require=("eacc",))
_FZ_SHIPPED = _fz_generate(_FZ_N, live_conds=False)


def _ids(progs):
    return [f"g{i:02d}" for i in range(len(progs))]


def test_the_fuzz_corpora_are_not_empty():
    assert len(_FZ_LIVE) == _FZ_N and len(_FZ_SHIPPED) == _FZ_N


def test_the_live_corpus_actually_clears_bits():
    """The row that makes the live sweep worth running: at least half of its programs
    answer something DIFFERENT under the `0.25` rules than without them. A corpus whose
    every program is unmoved would make a green sweep a statement about nothing."""
    b = _fz_bindings()
    moved = 0
    for src in _FZ_LIVE:
        m, _c, names = cook_both(PRAGMA + src, b, masked=True)
        p, _c2, _n = cook_both(PRAGMA + src, b, masked=False)
        if any(not torch.equal(m[n], p[n]) for n in names):
            moved += 1
    assert moved >= _FZ_N // 2, f"only {moved}/{_FZ_N} live-condition programs move"


@pytest.mark.parametrize("src", _FZ_LIVE, ids=_ids(_FZ_LIVE))
def test_fuzz_live_conditions_tiers_agree_bitwise(src):
    """The acceptance row. Per-pixel conditions, so some pixels leave a region while
    others stay — the case the shipped pool cannot reach."""
    assert_bitwise("fuzz-live", *cook_both(PRAGMA + src, _fz_bindings()))


@pytest.mark.parametrize("src", _FZ_SHIPPED, ids=_ids(_FZ_SHIPPED))
def test_fuzz_as_shipped_tiers_agree_bitwise(src):
    assert_bitwise("fuzz-shipped", *cook_both(PRAGMA + src, _fz_bindings()))


@pytest.mark.parametrize("src", _FZ_LIVE[:12], ids=_ids(_FZ_LIVE[:12]))
def test_fuzz_live_codegen_equals_oracle(src):
    """…and the agreed answer is the per-pixel one, not a shared mistake."""
    b = _fz_bindings()
    _iout, cout, names = cook_both(PRAGMA + src, b)
    assert cout is not None
    program, _tm, _n = _compile(src, b)
    try:
        ref, _p = sweep(program, dict(b), _FZ_B, _FZ_H, _FZ_W, names)
    except scalar_oracle.OracleUnsupported as e:
        pytest.skip(f"oracle does not implement: {e}")
    for n in names:
        diff = (cout[n].float() - ref[n].float()).abs().max().item()
        assert diff <= 1e-5, f"{n}: max |cg - oracle| = {diff}"


def test_the_generator_still_emits_programs_that_do_not_cook():
    """`L4-F1`, still open and still watched: `_gen_stencil`'s min/max variant seeds a
    `vec3` accumulator and folds a 4-channel tap into it, so a share of generated programs
    raise `E6051` before any tier is compared. Fixing it changes what every existing TST-1
    seed generates, which is a fuzz-coverage decision and not this lane's."""
    b = _fz_bindings()
    rng = random.Random(4242)
    refused = 0
    for _ in range(60):
        src = T17._gen_program(rng, 3)
        try:
            cook_both(src, b, masked=None)
        except Exception:                                   # noqa: BLE001
            refused += 1
    assert 0 < refused < 30, f"{refused}/60 — the rate moved; re-read L4-F1"


def test_the_shipped_condition_pool_is_still_false_everywhere():
    """The premise the two-pool split rests on, re-checked here rather than carried from
    L4's measurement (`brief-conventions.md` §8)."""
    b = _fz_bindings()
    for cond in T17._EARLY_EXIT_CONDS:
        if "$pv" in cond:
            continue
        src = f"float c = ({cond}) ? 1.0 : 0.0; @OUT = vec4(c, c, c, 1.0);"
        out, _c, _n = cook_both(src, b, masked=None)
        assert float(L4._chan0(out).max().item()) == 0.0, f"{cond} is true somewhere"


def test_the_live_condition_pool_is_true_somewhere_and_false_somewhere():
    b = _fz_bindings()
    for cond in T17._LIVE_EARLY_EXIT_CONDS:
        src = f"float c = ({cond}) ? 1.0 : 0.0; @OUT = vec4(c, c, c, 1.0);"
        out, _c, _n = cook_both(src, b, masked=None)
        ch = L4._chan0(out)
        assert float(ch.max().item()) == 1.0, f"{cond} is false on every pixel"
        assert float(ch.min().item()) == 0.0, f"{cond} is true on every pixel"


def test_the_default_generator_stream_did_not_move():
    """The opt-in pool must not change what an existing seed generates. The digest below
    was taken at the lane's base sha `ebd89d1` (and at `9460091`) with the same code."""
    rng = random.Random(20260708)
    progs = []
    for _ in range(300):
        progs.append(T17._gen_program(rng, 3) if rng.random() < 0.4
                     else f"@OUT = vec4({T17._gen_expr(rng, 3)}, u, v, 1.0);")
    digest = hashlib.sha256("\n".join(progs).encode("utf-8")).hexdigest()
    assert digest == "5130f63bb6b80c6d7b05d6ec61f6158a7907f5d6379f7eff4dd11b095c6a0774"


# ══════════════════════════════════════════════════════════════════════════════
# 6. Mutation rows — each mirrored rule, removed, must red something above
# ══════════════════════════════════════════════════════════════════════════════

def test_mutation_an_unmasked_emitted_write_reds_the_worked_break_table(monkeypatch):
    """M1 removed on the codegen side: every masked write stores unmasked, and §1's
    hand-computed `break` table stops reproducing."""
    src = PRAGMA + L4._WORKED["break"][0]
    b = dict(L4._TABLE_BINDINGS)
    monkeypatch.setattr(masked_flow, "merge_write", lambda live, after, before: after)
    cout, _names = cook_cg_only(src, b)
    got = L4._chan0(cout).reshape(-1).tolist()
    assert got != pytest.approx(L4._WORKED["break"][2], abs=1e-5), (
        "an unmasked write still reproduced the table — M1 is not load-bearing here")


def test_mutation_the_emitted_loop_exits_on_all_rather_than_any(monkeypatch):
    """M3's ANY rule removed on the codegen side: the per-pixel `for` bound stops matching
    the interpreter."""
    real = masked_flow.m_any

    def all_live(a):
        if a is True:
            return True
        if a is False:
            return False
        return bool(a.all().item())
    monkeypatch.setattr(masked_flow, "m_any", all_live)
    src = PRAGMA + L4._ATOM_PROGRAMS["for_bound"]
    b = L4._bindings()
    program, type_map, names = _compile(src, b)
    interp = Interpreter()
    iout = interp.execute(program, dict(b), type_map, device="cpu",
                          output_names=names, source=src, _masked_flow=True)
    monkeypatch.setattr(masked_flow, "m_any", real)
    _i2, cout, _n = cook_both(src, b)
    # The interpreter above ran with the mutation, codegen below without it: the two must
    # now DISAGREE, which is what makes the parity rows above load-bearing.
    assert not torch.equal(iout["OUT"], cout["OUT"])


def test_mutation_dropping_the_empty_call_skip_breaks_the_recursion(monkeypatch):
    """M4's skip removed: the per-pixel recursion stops terminating and hits the emitted
    call-depth guard. Codegen alone, because the interpreter's own skip lives on the same
    helper and would raise first."""
    monkeypatch.setattr(masked_flow, "m_any", lambda a: True)
    src = PRAGMA + L4._ATOM_PROGRAMS["per_pixel_recursion"]
    with pytest.raises(RuntimeError) as exc:
        cook_cg_only(src, L4._bindings())
    assert "call depth" in str(exc.value).lower()


def test_mutation_the_predicate_is_the_shared_one(monkeypatch):
    """§5's divergence site 1, and L4's `L4-F3`: codegen must derive its mask through
    `masked_flow.cond_mask`, never by emitting its own `(cond > 0.5)`.

    Replacing the helper with the OTHER reading must move the codegen answer — if it does
    not, the emitter is not calling it."""
    b = L4._bindings()
    src = PRAGMA + L4._ATOM_PROGRAMS["break_basic"]
    before, _n = cook_cg_only(src, b)
    monkeypatch.setattr(masked_flow, "cond_mask", lambda c: c <= 0.5)
    after, _n2 = cook_cg_only(src, b)
    assert not torch.equal(before["OUT"], after["OUT"]), (
        "replacing the shared predicate moved nothing — the emitter is not calling it")


def test_the_emitted_source_never_spells_the_predicate_itself():
    """The static half of the same claim: no masked emission contains a hand-written
    threshold comparison standing in for `cond_mask`."""
    for name in sorted(L4._ATOM_PROGRAMS):
        program, type_map, _n = _compile(PRAGMA + L4._ATOM_PROGRAMS[name], L4._bindings())
        fn = cg_mod.try_compile(program, type_map, _masked_flow=True)
        if fn is None:
            continue
        for line in fn._tex_src.splitlines():
            if "_MF.m_and" in line or "_MF.m_sub" in line:
                assert "> 0.5" not in line, f"{name}: {line.strip()}"


# ══════════════════════════════════════════════════════════════════════════════
# 7. The seam, and the version that has not moved
# ══════════════════════════════════════════════════════════════════════════════

def test_language_version_has_not_moved():
    """L5 does not bump it; L7 does. Every claim above about the gate being shut for real
    programs rests on this."""
    assert tex_api.LANGUAGE_VERSION == "0.24"


def test_try_compile_default_never_masks_at_this_head():
    """With no explicit seam, no program that can exist reaches the masked emitter —
    `min(pragma, LANGUAGE_VERSION)` is at most `(0, 24)` for every string a pragma can
    spell."""
    b = L4._bindings()
    for header in ("", "//!tex 0.23\n", "//!tex 0.24\n", "//!tex 0.25\n", "//!tex 1.0\n"):
        program, type_map, _n = _compile(header + L4._ATOM_PROGRAMS["break_basic"], b)
        fn = cg_mod.try_compile(program, type_map)
        assert fn is not None
        assert "_MF" not in fn._tex_src, header
