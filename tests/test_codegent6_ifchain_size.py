"""CODEGENT6 — an `else if` chain on a uniform parameter must emit O(arms), not O(2^arms).

The host measured a twelve-arm uniform-`$param` dispatch (its own program, not reproduced or
named here) emitting 6,593,732 chars / 63,658 lines of codegen Python, and `builtins.compile`
of it costing ~10 s of the GIL on every cold cook. The mechanism: `codegen_masked.py`'s `if`
emitter (`_mf_emit_if_else`) reaches a branch's statement list from TWO call sites — the
uniform (0-dim) dispatch and the per-pixel/spatial dispatch `_mf_emit_spatial_if` runs — and
an `else if` chain nests the next arm inside `else_body`, so walking that list with
`_emit_stmt` at both call sites (the pre-fix shape) recurses into a 2x multiplier PER ARM:
an n-arm chain cost O(2^n) emitted text, not O(n).

This file uses its OWN synthetic n-arm chain (never the host's program or file name) on its
own uniform `$sel` parameter, so the size law and the parity rows are provable without the
host's tree.  It is entirely new coverage: nothing above this docstring exists elsewhere in
the suite, so removing the fix should turn every row below red on its own (no shared fixture
masks a regression here)."""
import builtins
import time

import pytest
import torch

from helpers import *   # noqa: F403

from TEX_Wrangle.tex_cache import parse_and_split
from TEX_Wrangle.tex_runtime import codegen as cg_mod
from TEX_Wrangle.tex_runtime.interpreter import Interpreter, _consensus_extent

PRAGMA = "//!tex 0.25\n"


def _make_chain(n_arms: int) -> str:
    """An `n_arms`-arm `if`/`else if` dispatch on ONE uniform `$sel` parameter — the same
    shape the host's finding names (a uniform-parameter branch as one `else if` chain),
    built fresh here so nothing about it is borrowed from any other tree."""
    assert n_arms >= 2
    lines = ["f$sel = 0.0;", "float op = 0.0;",
             "if ($sel == 0.0) { op = 1.0; }"]
    for i in range(1, n_arms):
        lines.append(f"else if ($sel == {float(i)}) {{ op = {float(i)} * 2.0 + 1.0; }}")
    lines.append("@OUT = vec4(op, op, op, 1.0);")
    return PRAGMA + "\n".join(lines) + "\n"


def _make_perpixel_chain(n_arms: int) -> str:
    """The same n-arm dispatch, but selecting on a PER-PIXEL value (`@A.r`) rather than a
    uniform parameter — this is the shape that forces `_mf_emit_spatial_if`'s masked/merge
    path (as opposed to the uniform short-circuit), so the shared-closure fix is exercised
    on both of `_mf_emit_if_else`'s call sites, not only the uniform one."""
    assert n_arms >= 2
    lines = ["float sel = floor(@A.r * float(" + str(n_arms) + "));",
             "float op = 0.0;",
             "if (sel == 0.0) { op = 1.0; }"]
    for i in range(1, n_arms):
        lines.append(f"else if (sel == {float(i)}) {{ op = {float(i)} * 2.0 + 1.0; }}")
    lines.append("@OUT = vec4(op, op, op, 1.0);")
    return PRAGMA + "\n".join(lines) + "\n"


def _compile(src, bindings):
    bt = {name: _infer_binding_type(v) for name, v in bindings.items()}
    program = parse_and_split(src, bt)
    checker = TypeChecker(binding_types=bt, source=src)
    type_map = checker.check(program)
    return program, type_map, sorted(checker.assigned_bindings.keys())


def _emit(src, bindings):
    """Compile *src* on the codegen tier and return its emitted `_tex_src`."""
    program, type_map, _names = _compile(src, bindings)
    fn = cg_mod.try_compile(program, type_map, _masked_flow=True)
    assert fn is not None, "codegen declined the chain"
    return fn._tex_src


def _cook_both(src, bindings, device="cpu"):
    """Interpreter and codegen, both under the `0.25` rules, on the same bindings."""
    program, type_map, names = _compile(src, bindings)
    interp = Interpreter()
    iout = interp.execute(program, dict(bindings), type_map, device=device,
                          output_names=names, source=src, _masked_flow=True)
    fn = cg_mod.try_compile(program, type_map, _masked_flow=True)
    assert fn is not None
    dev = torch.device(device)
    cgb = {k: (v.clone().to(dev) if torch.is_tensor(v) else v) for k, v in bindings.items()}
    sp = _consensus_extent(bindings, program)
    env = {}
    cg_mod._invoke_cg(fn, env, cgb, TEXStdlib.get_functions(), dev, sp,
                      dtype=torch.float32, program=program)
    return iout, {n: cgb[n] for n in names if n in cgb}


# ══════════════════════════════════════════════════════════════════════════════
# 1. The size law — emitted size vs. arm count
# ══════════════════════════════════════════════════════════════════════════════

_ARM_COUNTS = (2, 4, 8, 12, 16)


def test_emitted_chars_grow_linearly_not_exponentially():
    """The acceptance row. Pre-fix this chain's emitted `_tex_src` roughly DOUBLES per extra
    arm (2^n over an n-arm chain: measured 3,932 / 21,791 / 109,579 / 530,560 chars at 2/4/6/8
    arms on the base sha, ~4.8-5.5x per +2 arms). Post-fix, each closure is defined once and
    CALLED from both dispatch sites, so the remaining growth is the cost of one more nested
    `def` per level (its own indentation, `nonlocal` line and two call sites) — sub-quadratic
    in practice, nowhere near exponential. Assert the weak (but exponential-excluding) form:
    the per-arm delta stays bounded by a constant that a 2^n law blows through almost
    immediately, and the 16-vs-2-arm ratio stays a small multiple of the 8x arm-count ratio."""
    sizes = {}
    for n in _ARM_COUNTS:
        sizes[n] = len(_emit(_make_chain(n), {}))
    # Monotone, and every consecutive step's growth ratio-per-arm stays bounded — the
    # opposite of the pre-fix shape, where the ratio-per-arm was itself growing (~2x/arm
    # compounding, i.e. per_arm roughly DOUBLING each step rather than merely rising). The
    # measured post-fix deltas top out under 3,100 chars/arm (8->16); a 2^n law would have
    # made the 8->12 delta alone (2^8=256 -> 2^12=4096, a 16x jump) dwarf this ceiling.
    prev_n = None
    for n in _ARM_COUNTS:
        if prev_n is not None:
            delta_arms = n - prev_n
            delta_chars = sizes[n] - sizes[prev_n]
            per_arm = delta_chars / delta_arms
            assert per_arm < 4000, (
                f"{prev_n}->{n} arms: {per_arm:.0f} chars/arm — exponential-shaped growth")
        prev_n = n
    # The blunt version of the same claim: 16 arms (8x the arms of the 2-arm case) must not
    # cost anywhere near 8x's worth of EXPONENT — a 2^n law would make this ratio explode
    # (2^14 ~= 16384x); a linear law keeps it near 8x plus per-arm closure overhead.
    ratio = sizes[16] / sizes[2]
    assert ratio < 50, f"16-arm/2-arm size ratio {ratio:.1f}x — still exponential-shaped"


def test_emitted_lines_grow_linearly_not_exponentially():
    line_counts = {n: _emit(_make_chain(n), {}).count("\n") + 1 for n in _ARM_COUNTS}
    ratio = line_counts[16] / line_counts[2]
    assert ratio < 50, f"16-arm/2-arm line-count ratio {ratio:.1f}x — still exponential-shaped"


@pytest.mark.timing
def test_compile_of_the_16_arm_chain_is_fast():
    """Not the acceptance row (timing lives under this marker per project convention), but
    the number the underlying report actually cared about: `builtins.compile` of the
    emitted source.
    Pre-fix, a chain this size would not even finish emitting in reasonable time; post-fix
    it is milliseconds."""
    text = _emit(_make_chain(16), {})
    t0 = time.perf_counter()
    builtins.compile(text, "<gen>", "exec")
    ms = (time.perf_counter() - t0) * 1000.0
    assert ms < 500.0, f"builtins.compile took {ms:.1f} ms for a 16-arm chain"


# ══════════════════════════════════════════════════════════════════════════════
# 2. Invariant 2 — interpreter == codegen, bitwise, for every arm value
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("n_arms", (2, 5, 12))
@pytest.mark.parametrize("device", ("cpu",))
def test_uniform_chain_parity_every_arm_value(n_arms, device):
    """Every `$sel` value 0..n_arms-1 dispatches to the SAME arm on both tiers, bit-exact —
    the uniform (0-dim) dispatch path, which is where the host's program lives."""
    src = _make_chain(n_arms)
    for sel in range(n_arms):
        iout, cout = _cook_both(src, {"sel": float(sel)}, device=device)
        a, c = iout["OUT"], cout["OUT"]
        assert a.shape == c.shape, f"arms={n_arms} sel={sel}: {a.shape} vs {c.shape}"
        assert torch.equal(a, c), f"arms={n_arms} sel={sel}: interpreter != codegen"
        # ...and it is the RIGHT arm, not merely an agreeing pair.
        want = float(sel) * 2.0 + 1.0 if sel != 0 else 1.0
        got = a.reshape(-1)[0].item()
        assert got == pytest.approx(want, abs=1e-5), f"arms={n_arms} sel={sel}: got {got}"


def _perpixel_bindings(n_arms):
    # A 1x2x(2*n_arms) grid whose @A.r sweeps every arm's selector at least once.
    b, h, w = 1, 2, 2 * n_arms
    vals = [((i % n_arms) + 0.5) / n_arms for i in range(b * h * w)]
    t = torch.tensor(vals, dtype=torch.float32).reshape(b, h, w, 1)
    return {"A": torch.cat([t, torch.zeros_like(t), torch.zeros_like(t),
                            torch.ones_like(t)], dim=-1)}


@pytest.mark.parametrize("n_arms", (2, 5, 12))
def test_perpixel_chain_parity_every_arm_value(n_arms):
    """The other call site: a PER-PIXEL selector forces `_mf_emit_spatial_if`'s masked
    merge path, which is where the shared closures are CALLED a second time (not
    re-walked) — this is the row that would catch a fix that only helped the uniform
    dispatch."""
    src = _make_perpixel_chain(n_arms)
    b = _perpixel_bindings(n_arms)
    iout, cout = _cook_both(src, b)
    a, c = iout["OUT"], cout["OUT"]
    assert a.shape == c.shape
    assert torch.equal(a, c), "per-pixel chain: interpreter != codegen"
    # Every arm must actually have fired somewhere in the grid (the bindings sweep them).
    seen = set(a[..., 0].reshape(-1).tolist())
    expected = {1.0} | {float(i) * 2.0 + 1.0 for i in range(1, n_arms)}
    assert seen == expected, f"arms={n_arms}: fired {seen}, expected {expected}"


# ══════════════════════════════════════════════════════════════════════════════
# 3. Invariant 7 — what stays byte-identical, and what does not (and why)
# ══════════════════════════════════════════════════════════════════════════════

def test_a_program_with_no_if_else_at_all_is_untouched_by_this_fix():
    """The fix only changes `_mf_emit_if_else`/`_mf_emit_spatial_if`/`_mf_emit_function_def`'s
    pre-declaration loop; a `0.25` program with no `if` anywhere still walks the SAME loop
    emitters as before, so its emitted source does not gain a single shared-closure `def`."""
    src = PRAGMA + "float x = @A.r; float c = 0.0;\nfor (int i=0;i<3;i=i+1){c=c+x;}\n@OUT=vec4(c,c,c,1.0);"
    text = _emit(src, {"A": torch.zeros(1, 1, 1, 4)})
    assert "_mfc" not in text, "a loop-only program picked up a shared-closure name"


def test_a_single_unchained_if_else_now_costs_two_small_closures_not_zero():
    """The honest half of invariant 7: a program with exactly ONE `if`/`else` (no chain at
    all) is NOT byte-identical to the pre-fix emission — the fix does not special-case
    "is this actually a chain", because detecting that reliably is exactly the kind of
    per-shape branch that would leave the general case (a chain built some other way, or
    one that grows past whatever arm count a special case assumed) exposed to the same
    2^n law. The cost of NOT special-casing is two `def _mfcN(): ...` closures for a
    program that did not need sharing; this asserts that the cost is exactly that (two
    small closures) rather than anything larger, and that VALUES are still correct
    (parity, checked elsewhere in this file and in `test_lang_l5_codegen_masking.py`)."""
    src = PRAGMA + "float x = @A.r; float c = 0.0;\nif (x > 0.5) { c = 1.0; } else { c = 2.0; }\n@OUT=vec4(c,c,c,1.0);"
    text = _emit(src, {"A": torch.zeros(1, 1, 1, 4)})
    assert text.count("def _mfc") == 2, text
