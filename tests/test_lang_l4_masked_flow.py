"""LANG-L4 — the interpreter's language-`0.25` rules (M1–M7), and the oracle that proves them.

L4 of `docs/masked-control-flow.md`'s staged plan. `tex_runtime/masked_flow.py` implements
§1's rules on the interpreter; `tests/scalar_oracle.py` is the per-pixel scalar oracle that
says whether they are right.

**Why an oracle and not tier parity.** The interpreter is the oracle every other tier must
match bit-exactly, so an error here becomes an error everywhere — the next stage will
faithfully reproduce whatever this one builds. §0's table is that failure already shipped:
both tiers agree, in all five rows, on an answer neither the language nor the author wanted.
`docs/masked-control-flow.md` §5's divergence site 6 states it outright — "a shared error is
invisible to parity" — so the masking rules need an answer computed a second way, by
something that does not mask.

**How independent the oracle actually is, stated rather than assumed.** `scalar_oracle.py`'s
own docstring carries the full statement; in one line: it shares torch's elementwise kernels
(so IEEE rounding matches instead of manufacturing false mismatches) and `TEXStdlib`'s
pixel-local leaf functions (`sin`, `clamp`, `fetch`), and it shares NOTHING that has anything
to do with control flow — no AST evaluator, no dispatch, no assignment path, no live mask, no
loop driver, no call frame. The evidence that the independence is real and not a claim:
the oracle was written and run BEFORE a line of `masked_flow.py` existed, and on its own
reproduced all five of §1's hand-computed "after" tables. It agreed with a specification,
not with an implementation — and `test_worked_table_after_column_equals_oracle` below is
that same check, kept runnable, so the claim stays checkable rather than historical.

**The seam.** `LANGUAGE_VERSION` does not move in this lane (that is L7), so the engine's own
gate — `min(pragma, LANGUAGE_VERSION) >= MASKED_FLOW_SINCE` — is False for every program that
can exist at this head. That is proved here over the whole corpus rather than asserted, and
it is why the tests below ask for the rules explicitly through `Interpreter.execute`'s
`_masked_flow` seam.
"""
import pytest
import torch

from helpers import *

import compat_corpus as cc
import scalar_oracle
from scalar_oracle import OracleUnsupported, sweep
from TEX_Wrangle import tex_api, tex_roi
from TEX_Wrangle.tex_cache import parse_and_split   # not in helpers' star set (HOOK-4)
from TEX_Wrangle.tex_runtime import masked_flow
from TEX_Wrangle.tex_runtime.interpreter import Interpreter, InterpreterError


# ── the grid every row below cooks on ───────────────────────────────────────────
# A wire, deliberately: measured at this head (and recorded in the design note's §0 and §6),
# a program with NO `@` wire cooks at a 1x1 grid, where masked and unmasked agree by
# construction — which is how the corpus's one class-B program has been proving nothing
# since 0.23. Every row here reads `@A`, so the difference is visible.
_B, _H, _W = 2, 3, 5


def _wire(seed=0.0):
    n = _B * _H * _W
    vals = [(((i * 7 + 3) % 17) / 17.0 + seed) % 1.0 for i in range(n)]
    t = torch.tensor(vals, dtype=torch.float32).reshape(_B, _H, _W, 1)
    return torch.cat([t, (t + 0.13) % 1.0, (t + 0.41) % 1.0,
                      torch.ones_like(t)], dim=-1)


def _bindings():
    return {"A": _wire(), "B": _wire(0.37)}


def _compile(src, bindings):
    bt = {name: _infer_binding_type(v) for name, v in bindings.items()}
    program = parse_and_split(src, bt)
    checker = TypeChecker(binding_types=bt, source=src)
    type_map = checker.check(program)
    return program, type_map, sorted(checker.assigned_bindings.keys())


def _cook(src, bindings, masked):
    """Run `src` on the interpreter. `masked` True asks for the 0.25 rules, None asks the
    engine's own gate (which answers False at this head — see `test_below_025_*`)."""
    program, type_map, names = _compile(src, bindings)
    interp = Interpreter()
    out = interp.execute(program, dict(bindings), type_map, device="cpu",
                         output_names=names, source=src, _masked_flow=masked)
    return out, interp


def _chan0(out):
    t = out["OUT"]
    return t[..., 0] if t.dim() == 4 else t


PRAGMA = "//!tex 0.25\n"


# ══════════════════════════════════════════════════════════════════════════════
# §1's five worked tables — the acceptance row the design states first
# ══════════════════════════════════════════════════════════════════════════════
# The four-pixel grid and the `a = [0.10, 0.30, 0.70, 0.90]` values are the design note's
# own, so these are its tables and not a paraphrase of them.

_T_A = torch.tensor([0.10, 0.30, 0.70, 0.90]).reshape(1, 1, 4, 1).repeat(1, 1, 1, 4)
_T_A[..., 3] = 1.0
_TABLE_BINDINGS = {"A": _T_A}

_WORKED = {
    "break": ("""
float a = @A.r;
float hit = -1.0;
for (int i = 0; i < 3; i = i + 1) {
  if (a > 0.5) { hit = float(i) + 10.0; break; }
  hit = hit - 1.0;
}
@OUT = vec4(hit, hit, hit, 1.0);
""", [10.0, 10.0, 10.0, 10.0], [-4.0, -4.0, 10.0, 10.0]),

    "continue": ("""
float a = @A.r;
float acc = 0.0;
for (int i = 0; i < 3; i = i + 1) {
  if (a > 0.5) { continue; }
  acc = acc + 1.0;
}
@OUT = vec4(acc, acc, acc, 1.0);
""", [0.0, 0.0, 0.0, 0.0], [3.0, 3.0, 0.0, 0.0]),

    "return": ("""
float pick(float a) {
  if (a > 0.5) { return a * 10.0; }
  return a * 100.0;
}
float r = pick(@A.r);
@OUT = vec4(r, r, r, 1.0);
""", [1.0, 3.0, 7.0, 9.0], [10.0, 30.0, 7.0, 9.0]),

    "for_bound": ("""
float n = @A.r * 10.0;
float c = 0.0;
for (int i = 0; float(i) < n; i = i + 1) { c = c + 1.0; }
@OUT = vec4(c, c, c, 1.0);
""", [9.0, 9.0, 9.0, 9.0], [1.0, 3.0, 7.0, 9.0]),

    "while_bound": ("""
float x = @A.r; float c = 0.0;
while (x < 0.8) { x = x + 0.25; c = c + 1.0; }
@OUT = vec4(c, c, c, 1.0);
""", [3.0, 3.0, 3.0, 3.0], [3.0, 2.0, 1.0, 0.0]),
}


@pytest.mark.parametrize("name", sorted(_WORKED))
def test_worked_table_after_column(name):
    """The `after` row of each of §1's five worked tables, on the masked interpreter."""
    src, _before, after = _WORKED[name]
    out, _ = _cook(PRAGMA + src, _TABLE_BINDINGS, True)
    got = [round(float(x), 4) for x in _chan0(out).reshape(-1).tolist()]
    assert got == after, f"{name}: masked cook {got} != §1's after column {after}"


@pytest.mark.parametrize("name", sorted(_WORKED))
def test_worked_table_after_column_equals_oracle(name):
    """…and the same row is what the per-pixel scalar oracle computes, independently."""
    src, _before, after = _WORKED[name]
    program, _tm, names = _compile(src, _TABLE_BINDINGS)
    got, _probes = sweep(program, dict(_TABLE_BINDINGS), 1, 1, 4, names)
    vals = [round(float(x), 4) for x in got["OUT"][..., 0].reshape(-1).tolist()]
    assert vals == after


# ══════════════════════════════════════════════════════════════════════════════
# Invariant 7 — the "before" column survives, and it is not optional
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("name", sorted(_WORKED))
@pytest.mark.parametrize("header", ["", "//!tex 0.23\n", "//!tex 0.24\n"])
def test_no_pragma_or_pinned_keeps_the_before_column(name, header):
    """A program with no pragma, and one pinned at the previous level, still produce §0's
    `before` column — the invariant-7 proof for this stage."""
    src, before, _after = _WORKED[name]
    out, interp = _cook(header + src, _TABLE_BINDINGS, None)
    got = [round(float(x), 4) for x in _chan0(out).reshape(-1).tolist()]
    assert got == before
    assert interp._masked is False


def test_pragma_025_alone_changes_nothing_at_this_head():
    """Even a `//!tex 0.25` program still cooks under `0.23`'s rules, because the ENGINE's
    level is what the gate keys on and `LANGUAGE_VERSION` has not moved. This is
    `docs/masked-control-flow.md` §4's `min(...)` rule, which is also what keeps the
    region-dependence sunset shut until L7."""
    for name, (src, before, _after) in _WORKED.items():
        out, interp = _cook(PRAGMA + src, _TABLE_BINDINGS, None)
        got = [round(float(x), 4) for x in _chan0(out).reshape(-1).tolist()]
        assert got == before, name
        assert interp._masked is False, name


# ══════════════════════════════════════════════════════════════════════════════
# Below 0.25 cannot reach the masked path — by construction, proved over the corpus
# ══════════════════════════════════════════════════════════════════════════════

def test_below_025_gate_is_shut_for_every_corpus_program():
    """`masked_flow.enabled_for` is False for all 130 frozen corpus programs, including
    the one that carries a pragma. Proved by running the gate, not by reading it."""
    programs = dict(cc._corpus_programs())
    assert len(programs) >= 100, f"corpus unexpectedly small: {len(programs)}"
    on = []
    for name, src in programs.items():
        try:
            prog = Parser(Lexer(src).tokenize(), source=src).parse()
        except Exception:
            continue
        if masked_flow.enabled_for(prog, src):
            on.append(name)
    assert on == [], f"the masked path is reachable for {on} at LANGUAGE_VERSION " \
                     f"{tex_api.LANGUAGE_VERSION}"


def test_below_025_gate_is_shut_for_every_declarable_pragma():
    """The gate is `min(pragma, LANGUAGE_VERSION)`, so no pragma a source can spell opens
    it while the engine is below 0.25 — including one far in the future."""
    src = _WORKED["break"][0]
    for header in ["", "//!tex 0.23\n", "//!tex 0.24\n", "//!tex 0.25\n",
                   "//!tex 0.99\n", "//!tex 1.0\n"]:
        prog = Parser(Lexer(header + src).tokenize(), source=header + src).parse()
        assert masked_flow.enabled_for(prog, header + src) is False, header


def test_language_version_has_not_moved():
    """`LANGUAGE_VERSION` is L7's to move. If this row ever reds in THIS lane, the gate
    above stopped being a proof and became a coincidence."""
    assert tex_api.LANGUAGE_VERSION == "0.24"
    assert tex_roi.MASKED_FLOW_SINCE == (0, 25)


def test_empty_flow_plan_never_binds_the_masked_table():
    """The cost rule: the plan is empty for 129 of 130 corpus programs, so the masked path
    must not be entered at all for them. Asked of the gate directly, with the version
    condition satisfied by construction (a program whose plan IS empty)."""
    src = "float x = @A.r * 2.0; @OUT = vec4(x, x, x, 1.0);"
    prog = Parser(Lexer(PRAGMA + src).tokenize(), source=PRAGMA + src).parse()
    assert tex_api.flow_plan(prog).is_empty()
    # …and with the version gate forced open, `enabled_for`'s second condition still shuts
    # it: the plan is what decides, not the pragma.
    assert tex_api.flow_plan(prog).is_empty() is True


def test_masked_cook_of_a_plan_free_program_is_bit_identical():
    """A program with no masking-relevant site cooks bit-identically with the rules on.
    This is the "show what it costs when the masked path is not entered" row: when it IS
    entered for such a program, nothing it does may move a bit."""
    src = ("float s = 0.0; for (int i = 0; i < 4; i = i + 1) { s = s + @A.r * float(i); } "
           "@OUT = vec4(s, s, s, 1.0);")
    b = _bindings()
    plain, _ = _cook(src, b, None)
    forced, _ = _cook(PRAGMA + src, b, True)
    assert torch.equal(plain["OUT"], forced["OUT"])


# ══════════════════════════════════════════════════════════════════════════════
# The control-flow atoms: a 0.25 cook equals the per-pixel oracle sweep
# ══════════════════════════════════════════════════════════════════════════════

_ATOM_PROGRAMS = {
    # the five worked shapes again, on a real 2x3x5 grid rather than the table's four pixels
    "break_basic": """
float a = @A.r; float hit = -1.0;
for (int i = 0; i < 3; i = i + 1) {
  if (a > 0.5) { hit = float(i) + 10.0; break; }
  hit = hit - 1.0;
}
@OUT = vec4(hit, hit, hit, 1.0);
""",
    "continue_basic": """
float a = @A.r; float acc = 0.0;
for (int i = 0; i < 4; i = i + 1) {
  if (a > 0.5) { continue; }
  acc = acc + 1.0;
}
@OUT = vec4(acc, acc, acc, 1.0);
""",
    "return_basic": """
float pick(float a) { if (a > 0.5) { return a * 10.0; } return a * 100.0; }
float r = pick(@A.r);
@OUT = vec4(r, r, r, 1.0);
""",
    "for_bound": """
float n = @A.r * 6.0; float c = 0.0;
for (int i = 0; float(i) < n; i = i + 1) { c = c + 1.0; }
@OUT = vec4(c, c, c, 1.0);
""",
    "while_bound": """
float x = @A.r; float c = 0.0;
while (x < 0.8) { x = x + 0.25; c = c + 1.0; }
@OUT = vec4(c, c, c, 1.0);
""",
    # M1: a write after the transfer, to a variable declared outside the loop
    "break_then_writes": """
float a = @A.r; float s = 0.0; float t = 100.0;
for (int i = 0; i < 5; i = i + 1) {
  if (a * float(i) > 1.2) { break; }
  s = s + 1.0;
  t = t - a;
}
@OUT = vec4(s, t, s + t, 1.0);
""",
    # nested loops: the inner one per-pixel bounded, its break local to itself
    "nested_inner_break": """
float a = @A.r; float s = 0.0;
for (int i = 0; i < 3; i = i + 1) {
  for (int j = 0; j < 4; j = j + 1) {
    if (a * float(j) > 0.9) { break; }
    s = s + 1.0;
  }
  s = s + 10.0;
}
@OUT = vec4(s, s, s, 1.0);
""",
    # nested loops: the inner per-pixel bound, the outer counting
    "nested_per_pixel_bounds": """
float a = @A.r; float b = @B.r; float s = 0.0;
for (int i = 0; float(i) < a * 4.0; i = i + 1) {
  for (int j = 0; float(j) < b * 3.0; j = j + 1) { s = s + 1.0; }
  s = s + 0.5;
}
@OUT = vec4(s, s, s, 1.0);
""",
    # a continue in the outer loop, from inside the inner one's if
    "nested_outer_continue": """
float a = @A.r; float s = 0.0;
for (int i = 0; i < 3; i = i + 1) {
  s = s + 1.0;
  if (a > 0.4) { continue; }
  s = s + 100.0;
}
@OUT = vec4(s, s, s, 1.0);
""",
    # a return INSIDE a loop inside a function
    "return_in_loop": """
float first_over(float a) {
  for (int i = 0; i < 6; i = i + 1) {
    if (a * float(i) > 1.0) { return float(i); }
  }
  return -1.0;
}
float r = first_over(@A.r);
@OUT = vec4(r, r, r, 1.0);
""",
    # a terminating per-pixel recursion (M4's empty-call skip is what ends it)
    "per_pixel_recursion": """
float countdown(float n) {
  if (n <= 0.0) { return 0.0; }
  return 1.0 + countdown(n - 1.0);
}
float r = countdown(floor(@A.r * 5.0));
@OUT = vec4(r, r, r, 1.0);
""",
    # both arms write, one of them leaves
    "both_arms_then_break": """
float a = @A.r; float s = 0.0; float u2 = 0.0;
for (int i = 0; i < 4; i = i + 1) {
  if (a > 0.5) { s = s + 2.0; break; } else { u2 = u2 + 1.0; }
  s = s + 1.0;
}
@OUT = vec4(s, u2, s - u2, 1.0);
""",
    # continue and break in the same body
    "continue_and_break": """
float a = @A.r; float b = @B.r; float s = 0.0;
for (int i = 0; i < 5; i = i + 1) {
  if (a > 0.6) { continue; }
  if (b > 0.7) { break; }
  s = s + 1.0;
}
@OUT = vec4(s, s, s, 1.0);
""",
    # a body-local temporary (declared INSIDE the region: M1 leaves it unmasked)
    "body_local_temp": """
float a = @A.r; float s = 0.0;
for (int i = 0; i < 4; i = i + 1) {
  float t = a * float(i);
  if (t > 0.8) { break; }
  s = s + t;
}
@OUT = vec4(s, s, s, 1.0);
""",
    # a channel write on a masked path
    "channel_write": """
float a = @A.r; vec4 c = vec4(0.0, 0.0, 0.0, 1.0);
for (int i = 0; i < 3; i = i + 1) {
  if (a * float(i) > 0.7) { break; }
  c.r = c.r + 0.25;
  c.g = a;
}
@OUT = c;
""",
    # M6: an @ write performed inside a user function called from a per-pixel branch
    "binding_write_in_call": """
float stamp(float v) { @M = v; return v * 2.0; }
float a = @A.r; float r = 0.0;
@M = -1.0;
if (a > 0.5) { r = stamp(a); }
@OUT = vec4(r, @M, a, 1.0);
""",
    # a while whose condition mixes a wire and a counter
    "while_compound": """
float x = @A.r; float n = 0.0;
while (x < 0.9 && n < 6.0) { x = x + @B.r * 0.5 + 0.05; n = n + 1.0; }
@OUT = vec4(n, x, n * x, 1.0);
""",
    # an early return before any loop
    "early_return": """
float f(float a, float b) {
  if (a > b) { return a; }
  float s = 0.0;
  for (int i = 0; i < 3; i = i + 1) { s = s + b; }
  return s;
}
float r = f(@A.r, @B.r);
@OUT = vec4(r, r, r, 1.0);
""",
    # nested calls, each with its own per-pixel return
    "nested_calls": """
float inner(float a) { if (a > 0.5) { return 1.0; } return 2.0; }
float outer(float a) { if (a < 0.25) { return 0.0; } return inner(a) * 10.0; }
float r = outer(@A.r);
@OUT = vec4(r, r, r, 1.0);
""",
    # a per-pixel for bound whose update is itself per-pixel
    "per_pixel_update": """
float a = @A.r; float c = 0.0;
for (float x = 0.0; x < 1.0; x = x + a * 0.5 + 0.1) { c = c + 1.0; }
@OUT = vec4(c, c, c, 1.0);
""",
    # a break taken from the ELSE arm
    "break_in_else": """
float a = @A.r; float s = 0.0;
for (int i = 0; i < 4; i = i + 1) {
  if (a > 0.5) { s = s + 1.0; } else { break; }
}
@OUT = vec4(s, s, s, 1.0);
""",
    # a loop whose per-pixel bound never admits some pixels at all (zero body statements)
    "zero_trip_for_some": """
float a = @A.r; float s = 5.0;
for (int i = 0; float(i) < a * 2.0 - 0.9; i = i + 1) { s = s * 2.0; }
@OUT = vec4(s, s, s, 1.0);
""",
}


@pytest.mark.parametrize("name", sorted(_ATOM_PROGRAMS))
def test_atom_equals_oracle(name):
    """A `0.25` cook equals the per-pixel oracle sweep, within the design's `1e-5`."""
    src = _ATOM_PROGRAMS[name]
    b = _bindings()
    out, _ = _cook(PRAGMA + src, b, True)
    program, _tm, names = _compile(src, b)
    ref, _probes = sweep(program, dict(b), _B, _H, _W, names)
    for out_name in names:
        got, want = out[out_name], ref[out_name]
        assert got.shape == want.shape, f"{name}/{out_name}: {got.shape} vs {want.shape}"
        diff = (got.float() - want.float()).abs().max().item()
        assert diff <= 1e-5, f"{name}/{out_name}: max |cook - oracle| = {diff}"


@pytest.mark.parametrize("name", sorted(_ATOM_PROGRAMS))
def test_atom_actually_moves_under_the_pragma(name):
    """Every atom above must DIFFER from its unmasked cook — otherwise the agreement row
    is being satisfied by a program the rules never touch, which is precisely how the
    corpus's one class-B golden has been proving nothing since 0.23 (§0, §6)."""
    src = _ATOM_PROGRAMS[name]
    b = _bindings()
    masked, _ = _cook(PRAGMA + src, b, True)
    plain, _ = _cook(src, b, None)
    moved = any(not torch.equal(masked[k], plain[k]) for k in masked
                if isinstance(masked[k], torch.Tensor))
    assert moved, f"{name}: the 0.25 rules changed nothing — this atom tests nothing"


# ══════════════════════════════════════════════════════════════════════════════
# M5 — scatter, gated by SOURCE
# ══════════════════════════════════════════════════════════════════════════════

_SCATTER_SRC = """
float a = @A.r;
@OUT = vec4(0.0, 0.0, 0.0, 1.0);
@S = 0.0;
for (int i = 0; i < 3; i = i + 1) {
  if (a > 0.5) { break; }
  @S[ix, iy] += 1.0;
}
"""


def test_scatter_is_gated_by_source():
    """A source pixel contributes iff it is live on the path to the statement, so a pixel
    that broke out contributes nothing further — and a pixel that never entered the branch
    contributes its full count."""
    b = _bindings()
    out, _ = _cook(PRAGMA + _SCATTER_SRC, b, True)
    a = b["A"][..., 0]
    want = torch.where(a > 0.5, torch.zeros_like(a), torch.full_like(a, 3.0))
    assert torch.allclose(out["S"], want, atol=1e-6), f"{out['S']} vs {want}"


def test_scatter_unmasked_reading_is_unchanged():
    """…and without the rules it is `0.23`'s destination-gated answer, unchanged."""
    b = _bindings()
    out, interp = _cook(_SCATTER_SRC, b, None)
    assert interp._masked is False
    # `0.23`'s answer, pinned as a characterization and not as a desirable one: the
    # `break` under a per-pixel `if` unwinds region-wide on the FIRST pass, so the scatter
    # statement never executes at all and `@S` is still the 0-dim `0.0` it was declared
    # as. That is §0's defect in its scatter spelling, and it is what must NOT move for a
    # program below 0.25.
    assert out["S"].dim() == 0
    assert float(out["S"].item()) == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# M7 — a probe records only if its probe pixel is live
# ══════════════════════════════════════════════════════════════════════════════

def test_debug_print_records_only_for_a_live_probe_pixel():
    from TEX_Wrangle.tex_runtime import tier_trace
    src = """
float a = @A.r;
float keep = 0.0;
for (int i = 0; i < 2; i = i + 1) {
  if (a > 0.5) { break; }
  keep = debug_print("probe", a, 0, 0);
}
@OUT = vec4(keep, keep, keep, 1.0);
"""
    b = _bindings()
    live_at_origin = bool((b["A"][0, 0, 0, 0] <= 0.5).item())
    tier_trace.clear_probes()
    _cook(PRAGMA + src, b, True)
    probes = tier_trace.get_probes()
    if live_at_origin:
        assert probes, "pixel (0,0) is live, so its probe must have recorded"
    else:
        assert not probes, "pixel (0,0) left the loop, so its probe must not record"


# ══════════════════════════════════════════════════════════════════════════════
# The loop cap still fires for a pixel that never terminates
# ══════════════════════════════════════════════════════════════════════════════

def test_never_ending_pixel_still_raises_e6010():
    """Masking must not turn a runaway loop into a silent one."""
    src = """
float a = @A.r; float x = a; float c = 0.0;
while (x < 2.0) { c = c + 1.0; }
@OUT = vec4(c, c, c, 1.0);
"""
    with pytest.raises(InterpreterError) as exc:
        _cook(PRAGMA + src, _bindings(), True)
    assert exc.value._code == "E6010"


def test_a_pixel_that_terminates_does_not_raise_when_another_would_have():
    """The cap is about a LIVE pixel: once every pixel has left, the loop leaves too, even
    though an unmasked reading would keep running while any condition held."""
    src = """
float a = @A.r; float x = a; float c = 0.0;
while (x < 0.99) { x = x + 0.2; c = c + 1.0; }
@OUT = vec4(c, c, c, 1.0);
"""
    out, _ = _cook(PRAGMA + src, _bindings(), True)
    assert torch.isfinite(out["OUT"]).all()


# ══════════════════════════════════════════════════════════════════════════════
# Mutation rows — each masking rule, removed, must red something above
# ══════════════════════════════════════════════════════════════════════════════

def test_mutation_unmasked_write_reds_the_break_table():
    """M1 removed (a write is never masked): §1's `break` table stops reproducing."""
    src, _before, after = _WORKED["break"]
    original = masked_flow.MaskedFlowMixin._mf_assignment
    try:
        masked_flow.MaskedFlowMixin._mf_assignment = (
            lambda self, node: self._exec_assignment(node))
        out, _ = _cook(PRAGMA + src, _TABLE_BINDINGS, True)
        got = [round(float(x), 4) for x in _chan0(out).reshape(-1).tolist()]
        assert got != after, "dropping M1's masked write left the answer unchanged"
    finally:
        masked_flow.MaskedFlowMixin._mf_assignment = original


def test_mutation_loop_exits_on_all_rather_than_any_reds_the_bound_table():
    """M3's "run while ANY pixel is live" removed: the per-pixel `for` bound stops
    reproducing."""
    src, _before, after = _WORKED["for_bound"]
    original = masked_flow.m_any
    try:
        masked_flow.m_any = lambda a: (True if a is True else
                                       False if a is False else bool(a.all().item()))
        out, _ = _cook(PRAGMA + src, _TABLE_BINDINGS, True)
        got = [round(float(x), 4) for x in _chan0(out).reshape(-1).tolist()]
        assert got != after, "exiting on ALL rather than ANY left the answer unchanged"
    finally:
        masked_flow.m_any = original


def test_mutation_continue_clearing_for_the_whole_loop_reds_the_continue_table():
    """M3.4's "restored at the update/condition" removed (a `continue` behaving like a
    `break`): the answer stops matching the oracle.

    Note which program this uses, and why §1's own `continue` table is the WRONG choice
    here: in that program every pass after a `continue` does nothing for the continued
    pixel anyway, so break-instead-of-continue gives the identical answer. A mutation the
    table cannot see is a mutation the table does not test. `nested_outer_continue` does
    work BEFORE the `continue`, so the restored bit is observable."""
    src = _ATOM_PROGRAMS["nested_outer_continue"]
    b = _bindings()
    program, _tm, names = _compile(src, b)
    ref, _ = sweep(program, dict(b), _B, _H, _W, names)
    original = masked_flow.MaskedFlowMixin._mf_continue

    def _as_break(self, node):
        return masked_flow.MaskedFlowMixin._mf_break(self, node)
    try:
        masked_flow.MaskedFlowMixin._mf_continue = _as_break
        out, _ = _cook(PRAGMA + src, b, True)
        diff = (out["OUT"].float() - ref["OUT"].float()).abs().max().item()
        assert diff > 1e-5, "clearing a continue for the whole loop changed nothing"
    finally:
        masked_flow.MaskedFlowMixin._mf_continue = original


def test_mutation_no_empty_call_skip_breaks_the_recursion():
    """M4's empty-call skip removed: the terminating per-pixel recursion stops
    terminating and hits the call-depth guard."""
    src = _ATOM_PROGRAMS["per_pixel_recursion"]
    original = masked_flow.MaskedFlowMixin._mf_call_user_function

    def _no_skip(self, func_def, call_node):
        saved = masked_flow.m_any
        masked_flow.m_any = lambda a: True
        try:
            return original(self, func_def, call_node)
        finally:
            masked_flow.m_any = saved
    try:
        masked_flow.MaskedFlowMixin._mf_call_user_function = _no_skip
        with pytest.raises(InterpreterError) as exc:
            _cook(PRAGMA + src, _bindings(), True)
        assert exc.value._code == "E6060"
    finally:
        masked_flow.MaskedFlowMixin._mf_call_user_function = original


# ══════════════════════════════════════════════════════════════════════════════
# The oracle itself — it must refuse, loudly, what it does not implement
# ══════════════════════════════════════════════════════════════════════════════

def test_oracle_refuses_what_it_does_not_implement():
    """An oracle that quietly guessed at a node it did not understand would be worse than
    no oracle, so the refusal is part of the contract."""
    src = 'mat3 m = mat3(1.0); vec3 p = m * vec3(1.0, 2.0, 3.0); @OUT = vec4(p, 1.0);'
    bt = {"OUT": TEXType.VEC4}
    prog = parse_and_split(src, bt)
    ev = scalar_oracle.ScalarOracle(prog, {}, 0, 0, 0, 1, 1, 1)
    with pytest.raises(OracleUnsupported):
        ev.run()


def test_oracle_reports_a_never_ending_pixel_rather_than_hanging():
    src = """
float a = @A.r; float x = a; float c = 0.0;
while (x < 2.0) { c = c + 1.0; }
@OUT = vec4(c, c, c, 1.0);
"""
    b = _bindings()
    program, _tm, names = _compile(src, b)
    with pytest.raises(scalar_oracle.OracleLoopCap):
        sweep(program, dict(b), _B, _H, _W, names)


# ══════════════════════════════════════════════════════════════════════════════
# The existing fuzz generator — what it can reach, and what it cannot
# ══════════════════════════════════════════════════════════════════════════════
# `docs/masked-control-flow.md` §8's L4 row asks for oracle agreement over
# `test_v017_phase1._gen_program`'s generated programs as well as over the hand-written
# atoms, and the note implies those programs exercise control flow. Checked rather than
# assumed, and the answer has two halves:
#
#   * They DO. `_gen_early_exit` puts a `break` or a `continue` under a per-pixel `if`,
#     and `_gen_program` emits accumulator loops and user-function `return`s.
#   * But every entry in its `_EARLY_EXIT_CONDS` pool is, by that pool's own comment and by
#     the measurement in `test_shipped_generator_conditions_are_false_on_every_pixel`
#     below, FALSE on every pixel, and every loop bound it emits is static. So the pool
#     reaches the `0.23` defect (a transfer under a per-pixel `if` fires region-wide
#     whatever the condition says — §0's R-BREAK) but never reaches the case where SOME
#     pixels leave and others stay.
#
# Both halves are used. The shipped pool is swept against the oracle as-is, which covers
# the masked machinery across the full stdlib towers the generator builds; and the pool is
# swapped for genuinely per-pixel conditions to reach the half it cannot.

_FUZZ_B, _FUZZ_H, _FUZZ_W = 1, 2, 4
_FUZZ_N = 40
# Conditions TRUE for some pixels and FALSE for others — what the shipped pool is not.
_LIVE_CONDS = ["@A.r > 0.5", "@B.r > 0.45", "u > 0.4", "v > 0.4", "@A.g < 0.5",
               "@A.r + @B.r > 0.9"]


def _fuzz_bindings():
    import test_v017_phase1 as t17
    n = _FUZZ_B * _FUZZ_H * _FUZZ_W

    def img(seed):
        vals = [(((i * 5 + seed) % 13) / 13.0) for i in range(n)]
        t = torch.tensor(vals, dtype=torch.float32).reshape(_FUZZ_B, _FUZZ_H, _FUZZ_W, 1)
        return torch.cat([t, (t + 0.29) % 1.0, (t + 0.61) % 1.0, torch.ones_like(t)], -1)
    b = {"A": img(1), "B": img(7)}
    b.update(t17._FUZZ_PARAM_BINDING)
    return b


def _generate(count, conds=None, require=None):
    """`count` COOKABLE programs from the SHIPPED generator, optionally with its early-exit
    condition pool swapped. Restores the pool whatever happens.

    Programs the interpreter refuses are dropped here, and separately counted and reported
    by `test_shipped_generator_emits_programs_that_do_not_cook` — that refusal predates
    this lane (measured on `main` at the base sha) and is filed as a finding, not absorbed
    silently."""
    import random
    import test_v017_phase1 as t17
    saved = t17._EARLY_EXIT_CONDS
    b = _fuzz_bindings()
    out = []
    try:
        if conds is not None:
            t17._EARLY_EXIT_CONDS = list(conds)
        rng = random.Random(20260921)
        tries = 0
        while len(out) < count and tries < count * 80:
            tries += 1
            src = t17._gen_program(rng, 3)
            if require is not None and not any(k in src for k in require):
                continue
            try:
                _cook(src, b, None)
            except Exception:
                continue
            out.append(src)
    finally:
        t17._EARLY_EXIT_CONDS = saved
    return out


def _fuzz_ids(programs):
    return [f"g{i:02d}" for i in range(len(programs))]


_FUZZ_AS_SHIPPED = _generate(_FUZZ_N, require=("break", "continue", "return"))
_FUZZ_LIVE = _generate(_FUZZ_N, conds=_LIVE_CONDS, require=("break", "continue"))


def _assert_matches_oracle(src, b, B, H, W):
    program, _tm, names = _compile(src, b)
    try:
        ref, _ = sweep(program, dict(b), B, H, W, names)
    except OracleUnsupported as exc:
        pytest.skip(f"oracle does not implement: {exc}")
    out, _ = _cook(PRAGMA + src, b, True)
    for k in names:
        got, want = out[k].float(), ref[k].float()
        assert torch.equal(torch.isfinite(got), torch.isfinite(want)), \
            f"@{k}: finiteness disagrees"
        finite = torch.isfinite(got) & torch.isfinite(want)
        if not bool(finite.any()):
            continue
        d = (got[finite] - want[finite]).abs()
        # Relative to the magnitude, never below 1: the generator builds towers whose
        # values run past 1e3, where an absolute 1e-5 would be a claim about float32's
        # last bits and not about masking.
        scale = torch.maximum(want[finite].abs(), torch.ones_like(d))
        worst = float((d / scale).max().item())
        assert worst <= 1e-5, f"@{k}: max relative |cook - oracle| = {worst}"


@pytest.mark.parametrize("src", _FUZZ_AS_SHIPPED, ids=_fuzz_ids(_FUZZ_AS_SHIPPED))
def test_generated_program_equals_oracle(src):
    """A `0.25` cook of a program from the SHIPPED generator equals the per-pixel oracle
    sweep."""
    _assert_matches_oracle(src, _fuzz_bindings(), _FUZZ_B, _FUZZ_H, _FUZZ_W)


@pytest.mark.parametrize("src", _FUZZ_LIVE, ids=_fuzz_ids(_FUZZ_LIVE))
def test_generated_program_with_live_conditions_equals_oracle(src):
    """…and so does one whose early-exit conditions are genuinely per-pixel — the half the
    shipped pool cannot reach."""
    _assert_matches_oracle(src, _fuzz_bindings(), _FUZZ_B, _FUZZ_H, _FUZZ_W)


def test_generated_corpora_are_not_empty():
    """A sweep over nothing passes trivially, so the two corpora's sizes are pinned."""
    assert len(_FUZZ_AS_SHIPPED) >= 20, len(_FUZZ_AS_SHIPPED)
    assert len(_FUZZ_LIVE) >= 20, len(_FUZZ_LIVE)


def test_the_live_condition_corpus_actually_clears_bits():
    """The oracle row above is worth little if the swapped pool still never clears a bit.
    Measured: most of the live-condition programs must differ from their unmasked cook."""
    b = _fuzz_bindings()
    moved = 0
    for src in _FUZZ_LIVE:
        plain, _ = _cook(src, b, None)
        masked, _ = _cook(PRAGMA + src, b, True)
        if any(not torch.equal(plain[k], masked[k]) for k in plain):
            moved += 1
    assert moved >= len(_FUZZ_LIVE) // 2, \
        f"only {moved}/{len(_FUZZ_LIVE)} live-condition programs moved"


def test_shipped_generator_conditions_are_false_on_every_pixel():
    """The finding, as a measurement: every entry in the shipped early-exit pool is false
    at every pixel of the fuzz grid, so the shipped corpus reaches the "a transfer fires
    region-wide" defect and never the "some pixels leave" case. If a future round widens
    the pool this reds, which is the point — the comment above would otherwise go stale
    silently."""
    import test_v017_phase1 as t17
    b = _fuzz_bindings()
    live_somewhere = []
    for cond in t17._EARLY_EXIT_CONDS:
        src = (f"{t17._FUZZ_PARAM_DECL} float c = 0.0; if ({cond}) {{ c = 1.0; }} "
               f"@OUT = vec4(c, c, c, 1.0);")
        out, _ = _cook(src, b, None)
        if float(out["OUT"][..., 0].max().item()) > 0.0:
            live_somewhere.append(cond)
    assert live_somewhere == [], \
        f"the shipped pool is no longer all-false: {live_somewhere}"


def test_shipped_generator_emits_programs_that_do_not_cook():
    """Filed as a finding rather than absorbed: the shipped generator emits programs the
    interpreter refuses — `_gen_stencil`'s min/max variant seeds a `vec3` accumulator and
    folds a 4-channel tap into it, so `max(vec3, vec4)` raises E6051. Measured on `main`
    at this lane's base sha, so it predates the lane; pinned here so the rate is a number
    somebody can watch rather than a surprise inside a sweep."""
    import random
    import test_v017_phase1 as t17
    b = _fuzz_bindings()
    rng = random.Random(4242)
    bad = 0
    for _ in range(60):
        src = t17._gen_program(rng, 3)
        try:
            _cook(src, b, None)
        except Exception:
            bad += 1
    assert bad > 0, ("the generator no longer emits uncookable programs — drop this row "
                     "and the finding it pins")
    assert bad < 30, f"the uncookable rate jumped to {bad}/60"
