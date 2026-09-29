"""Codegen emission bugs that served a different picture than the interpreter.

Invariant 2: the interpreter is the oracle and codegen matches it within 1e-5. Every row
here compiles through the engine pipeline a cook runs (type-check, optimize, re-check),
runs the interpreter and the codegen-only route on copies of the same bindings, and
requires the CODEGEN tier to have served the answer: a codegen crash that the route
quietly hands to the interpreter would otherwise compare the oracle against itself.

Each row's program is chosen so the optimizer cannot unroll or fold away the construct
under test (a `break`, a large trip count or a large body keeps a loop a loop).
"""
import re
import pytest
import torch

from TEX_Wrangle.tex_compiler.lexer import Lexer
from TEX_Wrangle.tex_compiler.parser import Parser
from TEX_Wrangle.tex_cache import get_cache
from TEX_Wrangle.tex_marshalling import infer_binding_type
from TEX_Wrangle.tex_runtime.interpreter import Interpreter
from TEX_Wrangle.tex_runtime.codegen import try_compile
from TEX_Wrangle.tex_runtime.compiled import _codegen_only_execute
from TEX_Wrangle.tex_runtime import tier_trace


def _img(B=1, H=4, W=4, C=4, seed=7):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(B, H, W, C, generator=g)


def _clone(bindings):
    return {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in bindings.items()}


def both_tiers(code, bindings):
    """(interpreter outputs, codegen outputs) for `code`; fails unless codegen served."""
    bt = {n: infer_binding_type(v) for n, v in bindings.items()}
    program = Parser(Lexer(code).tokenize(), source=code).parse()
    program, tm, _refs, assigned, _params, used = get_cache().compile_ast(
        program, bt, source=code)
    outs = sorted(assigned.keys())
    ref = Interpreter().execute(program, _clone(bindings), tm, device="cpu",
                                output_names=outs)
    assert try_compile(program, tm) is not None, "codegen declined the program"
    tier_trace.reset()
    got = _codegen_only_execute(program, _clone(bindings), tm, "cpu", output_names=outs,
                                used_builtins=used, fingerprint=None, time_context=None)
    rec = tier_trace.last()
    assert rec is not None and rec.tier == "codegen", (
        f"codegen did not serve: {None if rec is None else rec.reason}")
    return ref, got


def assert_parity(code, bindings, atol=1e-5):
    ref, got = both_tiers(code, bindings)
    for name, rv in ref.items():
        gv = got[name]
        assert tuple(rv.shape) == tuple(gv.shape), f"{name}: shape {tuple(gv.shape)} != {tuple(rv.shape)}"
        diff = (rv.float() - gv.float()).abs().max().item()
        assert diff <= atol, f"{name}: codegen differs from the interpreter by {diff}"
    return ref, got


# ── static-range for loops: the trip count is len(range(start, stop, step)) ──────────────
#
# A `break` keeps each loop out of the optimizer's unroller. `acc` counts passes, so the
# expected value is the interpreter's trip count.

_TRIP_ROWS = [
    ("step 3 over 31 (scalar body)",
     "float acc = 0.0;\nfor (int i = 0; i < 31; i += 3) { acc = acc + 1.0; if (acc > 500.0) { break; } }\n"
     "@OUT = vec3(acc) + @A.rgb * 0.0;", 11.0),
    ("step 2 over <= 20 (scalar body)",
     "float acc = 0.0;\nfor (int i = 0; i <= 20; i += 2) { acc = acc + 1.0; if (acc > 500.0) { break; } }\n"
     "@OUT = vec3(acc) + @A.rgb * 0.0;", 11.0),
    ("empty range, start above stop (scalar body)",
     "float acc = 0.0;\nfor (int i = 5; i < 3; i++) { acc = acc + 1.0; if (acc > 500.0) { break; } }\n"
     "@OUT = vec3(acc) + @A.rgb * 0.0;", 0.0),
    # A step that counts away from a `<` bound never reaches it: the loop runs until its
    # own `break` (C semantics), not zero times as a Python range would say.
    ("step against the bound runs to its break (scalar body)",
     "float acc = 0.0;\nfor (int i = 0; i < 10; i -= 1) { acc = acc + 1.0; if (acc > 500.0) { break; } }\n"
     "@OUT = vec3(acc) + @A.rgb * 0.0;", 501.0),
    ("step 3 over 10 (tensor body)",
     "vec3 s = vec3(0.0);\nfor (int i = 0; i < 10; i += 3) { s = s + @A.rgb * 0.0 + vec3(1.0); if (i > 500) { break; } }\n"
     "@OUT = s;", 4.0),
    ("empty range (tensor body)",
     "vec3 s = vec3(0.0);\nfor (int i = 10; i < 5; i++) { s = s + @A.rgb * 0.0 + vec3(1.0); if (i > 500) { break; } }\n"
     "@OUT = s;", 0.0),
    ("step 2 over 5, masked 0.25 program",
     "//!tex 0.25\nvec3 s = vec3(0.0);\nfor (int i = 0; i < 5; i += 2) { if (@A.r > 2.0) { break; } s = s + vec3(1.0); }\n"
     "@OUT = s + @A.rgb * 0.0;", 3.0),
    ("negative step over a <= bound, masked 0.25 program",
     "//!tex 0.25\nvec3 s = vec3(0.0);\nfor (int i = -4; i <= 4; i += 2) { if (@A.r > 2.0) { break; } s = s + vec3(1.0); }\n"
     "@OUT = s + @A.rgb * 0.0;", 5.0),
]


@pytest.mark.parametrize("label,code,want", _TRIP_ROWS, ids=[r[0] for r in _TRIP_ROWS])
def test_static_for_trip_count_matches_interpreter(label, code, want):
    ref, got = both_tiers(code, {"A": _img()})
    r = ref["OUT"][..., 0].unique().tolist()
    assert r == [want], f"oracle reads {r}, row expects {want}"
    assert torch.equal(ref["OUT"], got["OUT"]), (
        f"codegen ran {got['OUT'][..., 0].unique().tolist()} passes, interpreter {r}")


# ── general for loops: the literal bound is compared on the real counter value ─────────
#
# A fractional init or step keeps each loop off the static-range path; the fast bound test
# used to truncate both the counter and the literal with int().

_BOUND_ROWS = [
    ("t <= 1.0 stepping 0.25",
     "float acc = 0.0;\nfor (float t = 0.0; t <= 1.0; t += 0.25) { acc = acc + 1.0; if (acc > 500.0) { break; } }\n"
     "@OUT = vec3(acc) + @A.rgb * 0.0;", 5.0),
    ("negative counter, t < 0.0",
     "float acc = 0.0;\nfor (float t = -1.0; t < 0.0; t += 0.25) { acc = acc + 1.0; if (acc > 500.0) { break; } }\n"
     "@OUT = vec3(acc) + @A.rgb * 0.0;", 4.0),
    ("fractional literal bound x < 2.5",
     "float acc = 0.0;\nfor (float x = 0.0; x < 2.5; x += 1.0) { acc = acc + 1.0; if (acc > 500.0) { break; } }\n"
     "@OUT = vec3(acc) + @A.rgb * 0.0;", 3.0),
    ("y from -0.5 below 0.5",
     "float acc = 0.0;\nfor (float y = -0.5; y < 0.5; y += 0.25) { acc = acc + 1.0; if (acc > 500.0) { break; } }\n"
     "@OUT = vec3(acc) + @A.rgb * 0.0;", 4.0),
    ("fp32 accumulation against a fractional bound",
     "float acc = 0.0;\nfor (float t = 0.0; t <= 0.3; t += 0.1) { acc = acc + 1.0; if (acc > 500.0) { break; } }\n"
     "@OUT = vec3(acc) + @A.rgb * 0.0;", None),
    ("tensor body, t <= 1.0 stepping 0.25",
     "vec3 s = vec3(0.0);\nfor (float t = 0.0; t <= 1.0; t += 0.25) { s = s + @A.rgb * 0.0 + vec3(1.0); if (t > 500.0) { break; } }\n"
     "@OUT = s;", 5.0),
]


@pytest.mark.parametrize("label,code,want", _BOUND_ROWS, ids=[r[0] for r in _BOUND_ROWS])
def test_general_for_bound_matches_interpreter(label, code, want):
    ref, got = both_tiers(code, {"A": _img()})
    r = ref["OUT"][..., 0].unique().tolist()
    if want is not None:
        assert r == [want], f"oracle reads {r}, row expects {want}"
    assert torch.equal(ref["OUT"], got["OUT"]), (
        f"codegen ran {got['OUT'][..., 0].unique().tolist()} passes, interpreter {r}")


# ── while loops: the iteration cap fires only when the budget is spent ─────────────────
#
# The interpreter raises only when 1024 bodies ran without the loop ending; a loop that ends
# on its 1024th condition check, or breaks in its 1024th body, returns normally.

_WHILE_CAP_ROWS = [
    ("condition false on the 1024th check",
     "float acc = 0.0;\nint i = 0;\nwhile (i < 1023) { i = i + 1; acc = acc + 1.0; }\n"
     "@OUT = vec3(acc) + @A.rgb * 0.0;", 1023.0),
    ("break in the 1024th body",
     "float acc = 0.0;\nint i = 0;\nwhile (acc > -1.0) { i = i + 1; acc = acc + 1.0; if (i >= 1024) { break; } }\n"
     "@OUT = vec3(acc) + @A.rgb * 0.0;", 1024.0),
]


@pytest.mark.parametrize("label,code,want", _WHILE_CAP_ROWS, ids=[r[0] for r in _WHILE_CAP_ROWS])
def test_while_iteration_cap_matches_interpreter(label, code, want):
    ref, got = both_tiers(code, {"A": _img()})
    assert ref["OUT"][..., 0].unique().tolist() == [want]
    assert torch.equal(ref["OUT"], got["OUT"])


def test_while_runaway_still_raises_in_codegen():
    """The cap itself stays: 1024 completed bodies raise in the generated code too."""
    code = ("float acc = 0.0;\nwhile (acc > -1.0) { acc = acc + 1.0; }\n"
            "@OUT = vec3(acc) + @A.rgb * 0.0;")
    bt = {"A": infer_binding_type(_img())}
    program = Parser(Lexer(code).tokenize(), source=code).parse()
    program, tm, *_ = get_cache().compile_ast(program, bt, source=code)
    fn = try_compile(program, tm)
    assert fn is not None
    tier_trace.reset()
    with pytest.raises(Exception):
        Interpreter().execute(program, {"A": _img()}, tm, device="cpu", output_names=["OUT"])
    from TEX_Wrangle.tex_runtime.codegen import _invoke_cg
    from TEX_Wrangle.tex_runtime.stdlib import TEXStdlib
    with pytest.raises(RuntimeError, match="maximum iteration limit"):
        _invoke_cg(fn, {"__tex_scale": 1.0}, {"A": _img()}, TEXStdlib.get_functions(),
                   torch.device("cpu"), (1, 4, 4), program=program)


# ── copy-on-write across the re-emitted bodies of an if/else ──────────────────────────
#
# Each branch body is emitted more than once (the uniform path, the per-pixel path, and
# then/else in each). Ownership a write claimed in one emission does not hold in the next,
# so a channel write there must still clone rather than write through an alias.

_COW_ROWS = [
    ("per-pixel if writes a channel of an aliased vec4",
     "vec4 c = @A;\nvec4 d = c;\nif (u > 0.5) { c.r = 0.9; }\n@OUT = d + c * 0.0;", {}),
    ("uniform if/else writes different channels of an aliased vec4",
     "f$f = 0.0;\nvec4 p = @A;\nvec4 q = p;\nif ($f > 0.5) { p.r = 1.0; } else { p.g = 2.0; }\n"
     "@OUT = q + p * 0.0;", {"f": 0.0}),
    ("per-pixel if/else writes different channels of an aliased vec4",
     "vec4 p = @A;\nvec4 q = p;\nif (u > 0.5) { p.r = 1.0; } else { p.g = 2.0; }\n"
     "@OUT = vec4(q.r, q.g, p.r, p.g);", {}),
    ("the input binding itself is not written through",
     "vec4 c = @A;\nif (u > 0.5) { c.b = 0.25; } else { c.a = 0.5; }\n@OUT = @A + c * 0.0;", {}),
]


@pytest.mark.parametrize("label,code,extra", _COW_ROWS, ids=[r[0] for r in _COW_ROWS])
def test_if_else_channel_write_never_aliases(label, code, extra):
    a = _img()
    ref, got = assert_parity(code, {"A": a, **extra})
    assert "OUT" in ref


# ── a loop-carried variable is not its declaration's initializer ─────────────────────
#
# `sample(@A, x, v)` with `x` declared `= u` folds to a direct pixel fetch. Inside a loop
# that advances `x` after the read, the second pass reads a shifted coordinate, so the
# fold is wrong from the first reassignment on, whatever the textual order.

_LOOP_INIT_ROWS = [
    ("while loop advancing x after the sample",
     "float x = u;\nvec3 acc = vec3(0.0);\n"
     "for (int k = 0; k < 2; k++) {\n"
     "  float pin = ix + iy;\n"
     "  int i = 0;\n"
     "  while (i < 3) { acc = acc + sample(@A, x, v).rgb; x = x + 0.25; i = i + 1; }\n"
     "  if (pin < -1.0) { break; }\n"
     "}\n@OUT = acc;"),
    ("general for loop advancing x after the sample",
     "float x = u;\nvec3 acc = vec3(0.0);\n"
     "for (int k = 0; k < 2; k++) {\n"
     "  float pin = ix + iy;\n"
     "  for (float t = 0.5; t < 3.0; t += 1.0) { acc = acc + sample(@A, x, v).rgb; x = x + 0.25; }\n"
     "  if (pin < -1.0) { break; }\n"
     "}\n@OUT = acc;"),
]


@pytest.mark.parametrize("label,code", _LOOP_INIT_ROWS, ids=[r[0] for r in _LOOP_INIT_ROWS])
def test_loop_carried_var_is_not_folded_to_its_initializer(label, code):
    assert_parity(code, {"A": _img(H=6, W=6)})


def test_loop_invariant_initializer_keeps_the_direct_fetch():
    """Control: a name the loop never reassigns still folds (the fast path is kept)."""
    code = ("float x = u;\nvec3 acc = vec3(0.0);\n"
            "for (int k = 0; k < 2; k++) {\n"
            "  float pin = ix + iy;\n"
            "  int i = 0;\n"
            "  while (i < 3) { acc = acc + sample(@A, x, v).rgb; i = i + 1; }\n"
            "  if (pin < -1.0) { break; }\n"
            "}\n@OUT = acc;")
    bindings = {"A": _img(H=6, W=6)}
    assert_parity(code, bindings)
    bt = {"A": infer_binding_type(bindings["A"])}
    program = Parser(Lexer(code).tokenize(), source=code).parse()
    program, tm, *_ = get_cache().compile_ast(program, bt, source=code)
    src = try_compile(program, tm)._tex_src
    # The direct fetch clamps then casts, NaN-safe: `...shape[2] - 1)[.nan_to_num_(0.0)].long()`.
    assert re.search(r"shape\[2\] - 1\)(\.nan_to_num_\(0\.0\))?\.long\(\)", src), \
        "the zero-offset direct fetch was not emitted"


# ── a scatter write replaces the binding the loop's hoisted sample view was taken from ─
#
# The first scatter into a binding this cook does not own clones it (copy-on-write), so a
# BCHW view hoisted above the loop keeps reading the original data and misses every write.

_SCATTER_SAMPLE_ROWS = [
    ("while loop scatters then samples the same input",
     "vec3 acc = vec3(0.0);\nint i = 0;\n"
     "while (i < 4) { @A[i, 1] = vec4(1.0); acc = acc + sample(@A, u, v).rgb; i = i + 1; }\n"
     "@OUT = acc;"),
    ("general for loop scatters then samples the same input",
     "vec3 acc = vec3(0.0);\n"
     "for (float t = 0.5; t < 4.0; t += 1.0) { @A[t, 2] = vec4(0.0); acc = acc + sample(@A, u, v).rgb; }\n"
     "@OUT = acc;"),
]


@pytest.mark.parametrize("label,code", _SCATTER_SAMPLE_ROWS, ids=[r[0] for r in _SCATTER_SAMPLE_ROWS])
def test_sample_sees_scatter_writes_in_the_same_loop(label, code):
    assert_parity(code, {"A": _img(H=5, W=5)})


# ── emit-time state scoped to a user function stays inside it ─────────────────────────

_FN_SCOPE_ROWS = [
    ("a sample hoist made inside a function is not reused after it",
     "float f(float k) {\n  vec3 s = vec3(0.0);\n  int i = 0;\n"
     "  while (i < 2) { s = s + sample(@A, u, v).rgb; i = i + 1; }\n  return s.r * k;\n}\n"
     "float a = f(1.0);\nvec3 acc = vec3(0.0);\nint j = 0;\n"
     "while (j < 2) { acc = acc + sample(@A, u, v).rgb; j = j + 1; }\n"
     "@OUT = acc + vec3(a);"),
    ("a function-local initializer does not shadow the outer one",
     "float x = u + 0.1;\nfloat g() { float x = u; return x; }\nvec3 acc = vec3(0.0);\n"
     "for (int k = 0; k < 2; k++) {\n  float pin = ix + iy;\n  int i = 0;\n"
     "  while (i < 3) { acc = acc + sample(@A, x, v).rgb; i = i + 1; }\n"
     "  if (pin < -1.0) { break; }\n}\n@OUT = acc + vec3(g()) * 0.0;"),
]


@pytest.mark.parametrize("label,code", _FN_SCOPE_ROWS, ids=[r[0] for r in _FN_SCOPE_ROWS])
def test_function_scoped_emit_state_does_not_leak(label, code):
    assert_parity(code, {"A": _img(H=6, W=6)})


def test_scalar_loop_branch_local_that_never_ran():
    """A local declared in a branch the loop never took is still None after the loop."""
    code = ("float acc = 0.0;\n"
            "for (int i = 0; i < 12; i++) { if (i > 100) { float t = 2.0; acc = acc + t; } acc = acc + 1.0; }\n"
            "@OUT = vec3(acc) + @A.rgb * 0.0;")
    ref, got = assert_parity(code, {"A": _img()})
    assert ref["OUT"][..., 0].unique().tolist() == [12.0]


# ── stencil lowering claims only the nests and clusters it computes exactly ─────────────
#
# A lowering replaces a whole loop nest (or an inline tap cluster) with one pool, unfold
# or conv2d, so anything in it the pattern does not account for simply stops happening.
# Radius 5 keeps each nest past the optimizer's unroller.

_NEST = ("for (int dy = -5; dy <= 5; dy++) {{\n  for (int dx = -5; dx <= 5; dx++) {{\n{body}  }}\n}}\n")

_STENCIL_ROWS = [
    ("a tap temporary updated in the body (min/max)",
     "vec3 m = vec3(-10.0);\n" + _NEST.format(
         body="    vec3 s = fetch(@A, ix + dx, iy + dy).rgb;\n    s = s + vec3(0.1);\n    m = max(m, s);\n")
     + "@OUT = m;"),
    ("a tap temporary updated in the body (array collect)",
     "vec4 r[121];\nint k = 0;\n" + _NEST.format(
         body="    vec4 s = fetch(@A, ix + dx, iy + dy);\n    s = s + vec4(0.1);\n    r[k] = s;\n    k++;\n")
     + "@OUT = r[0] + r[60] + r[120];"),
    ("a counter beside a min/max accumulator",
     "vec3 m = vec3(-10.0);\nfloat n = 0.0;\n" + _NEST.format(
         body="    m = max(m, fetch(@A, ix + dx, iy + dy).rgb);\n    n = n + 1.0;\n")
     + "@OUT = m + vec3(n);"),
    ("two counters beside a box sum",
     "vec3 acc = vec3(0.0);\nfloat n = 0.0;\nfloat c = 0.0;\n" + _NEST.format(
         body="    acc = acc + fetch(@A, ix + dx, iy + dy).rgb;\n    n = n + 1.0;\n    c = c + 1.0;\n")
     + "@OUT = acc / n + vec3(c);"),
    ("a fractional literal radius on a float counter",
     "vec3 acc = vec3(0.0);\n"
     "for (float dy = -1.5; dy <= 1.5; dy += 1.0) {\n  for (float dx = -1.5; dx <= 1.5; dx += 1.0) {\n"
     "    acc = acc + fetch(@A, ix + dx, iy + dy).rgb;\n  }\n}\n@OUT = acc;"),
    ("an inline cluster with a fractional fetch offset",
     "vec4 a = fetch(@A, ix - 1.5, iy);\nvec4 b = fetch(@A, ix, iy);\nvec4 c = fetch(@A, ix + 1, iy);\n"
     "vec4 o = a * 0.25 + b * 0.5 + c * 0.25;\n@OUT = o;"),
    ("an inline cluster mixing a pixel read with sample() taps",
     "vec4 c = @A;\nvec4 l = sample(@A, u - px, v);\nvec4 r = sample(@A, u + px, v);\n"
     "vec4 o = c * 0.5 + l * 0.25 + r * 0.25;\n@OUT = o;"),
]


@pytest.mark.parametrize("label,code", _STENCIL_ROWS, ids=[r[0] for r in _STENCIL_ROWS])
def test_stencil_lowering_matches_interpreter(label, code):
    assert_parity(code, {"A": _img(H=12, W=12)})


# ── the median/array-collect lowering fills the array exactly as the loop would ────────
#
# The loop writes arr[k] with k counting from its seed, clamped into the DECLARED array;
# the unfold must leave the same array behind, whatever the declared size.

_MEDIAN_ROWS = [
    ("runtime radius smaller than the declared array",
     "i$r = 1;\nfloat m[25];\nint k = 0;\n"
     "for (int dy = -$r; dy <= $r; dy++) {\n  for (int dx = -$r; dx <= $r; dx++) {\n"
     "    vec3 s = fetch(@A, ix + dx, iy + dy);\n    m[k] = s.r;\n    k++;\n  }\n}\n"
     "@OUT = vec3(median(m), arr_sum(m), float(k));", {"r": 1}),
    ("literal radius, declared array larger than the tap count",
     "float m[130];\nint k = 0;\n" + _NEST.format(
         body="    vec3 s = fetch(@A, ix + dx, iy + dy);\n    m[k] = s.r;\n    k++;\n")
     + "@OUT = vec3(median(m), m[125], float(k));", {}),
    ("literal radius, declared array smaller than the tap count",
     "float m[100];\nint k = 0;\n" + _NEST.format(
         body="    vec3 s = fetch(@A, ix + dx, iy + dy);\n    m[k] = s.r;\n    k++;\n")
     + "@OUT = vec3(median(m), m[99], float(k));", {}),
    ("vec array smaller than the tap count",
     "vec4 m[100];\nint k = 0;\n" + _NEST.format(
         body="    m[k] = fetch(@A, ix + dx, iy + dy);\n    k++;\n")
     + "@OUT = m[99] + m[3];", {}),
    ("a constant index",
     "float m[121];\nint k = 0;\n" + _NEST.format(
         body="    vec3 s = fetch(@A, ix + dx, iy + dy);\n    m[0] = s.r;\n    k++;\n")
     + "@OUT = vec3(m[0], m[1], float(k));", {}),
    ("the counter bumped before the collect",
     "float m[121];\nint k = 0;\n" + _NEST.format(
         body="    vec3 s = fetch(@A, ix + dx, iy + dy);\n    k++;\n    m[k] = s.r;\n")
     + "@OUT = vec3(m[0], m[120], float(k));", {}),
    ("a counter seeded at one",
     "float m[121];\nint k = 1;\n" + _NEST.format(
         body="    vec3 s = fetch(@A, ix + dx, iy + dy);\n    m[k] = s.r;\n    k++;\n")
     + "@OUT = vec3(m[0], m[120], float(k));", {}),
    ("a box-sum counter seeded at one",
     "vec3 acc = vec3(0.0);\nfloat n = 1.0;\n" + _NEST.format(
         body="    acc = acc + fetch(@A, ix + dx, iy + dy).rgb;\n    n = n + 1.0;\n")
     + "@OUT = acc / n;", {}),
]


@pytest.mark.parametrize("label,code,extra", _MEDIAN_ROWS, ids=[r[0] for r in _MEDIAN_ROWS])
def test_array_collect_lowering_matches_interpreter(label, code, extra):
    assert_parity(code, {"A": _img(H=12, W=12), **extra})


def test_array_collect_lowering_is_kept_where_exact():
    """The first four rows still take the unfold (now filling the declared array); the
    three whose slots the unfold cannot place run the loop."""
    def src(code, extra):
        bindings = {"A": _img(H=12, W=12), **extra}
        bt = {n: infer_binding_type(v) for n, v in bindings.items()}
        program = Parser(Lexer(code).tokenize(), source=code).parse()
        program, tm, *_ = get_cache().compile_ast(program, bt, source=code)
        return try_compile(program, tm)._tex_src
    lowered = {label: ".unfold(" in src(code, extra) for label, code, extra in _MEDIAN_ROWS[:7]}
    assert lowered == {
        "runtime radius smaller than the declared array": True,
        "literal radius, declared array larger than the tap count": True,
        "literal radius, declared array smaller than the tap count": True,
        "vec array smaller than the tap count": True,
        "a constant index": False,
        "the counter bumped before the collect": False,
        "a counter seeded at one": False,
    }


# ── masked (0.25) if-arm closures: a for-loop header writes through to the hoisted local ──
#
# Each arm body is emitted once into a `def` both dispatch paths call; a name the arm's
# for-header assigns must be declared `nonlocal` there like any other write.

_ARM_HEADER_ROWS = [
    ("for-init assignment inside a per-pixel arm",
     "//!tex 0.25\nint k = 0; float acc = 0.0;\n"
     "for (int j = 0; j < 20; j++) { if (@A.g > 2.0) { break; } }\n"
     "if (u > 0.5) { for (k = 0; k < 4; k = k + 1) { acc = acc + 1.0; } }\n"
     "@OUT = vec3(acc, float(k), 0.0);"),
    ("the same arm inside a masked loop pass",
     "//!tex 0.25\nint k = 0; float acc = 0.0;\n"
     "for (int j = 0; j < 20; j++) { if (@A.g > 2.0) { break; }\n"
     "  if (u > 0.5) { for (k = 0; k < 4; k = k + 1) { acc = acc + 1.0; } } }\n"
     "@OUT = vec3(acc, float(k), 0.0);"),
]


@pytest.mark.parametrize("label,code", _ARM_HEADER_ROWS, ids=[r[0] for r in _ARM_HEADER_ROWS])
def test_masked_arm_for_header_writes_the_outer_variable(label, code):
    ref, _ = assert_parity(code, {"A": _img()})
    assert ref["OUT"][..., 1].max().item() == 4.0  # the arm ran and left k at its bound


# ── min/max: the interpreter's own maximum/minimum, operand emitted once ──────────────
#
# On a tie clamp keeps x while maximum/minimum pick an operand by kernel and layout, so a
# clamp shortcut can return the other signed zero; atan2(0, ±0) turns that into 0 against pi.

def _signed_zeros():
    z = torch.zeros(1, 4, 4, 4)
    z[..., 0] = -0.0
    return z


_MINMAX_ROWS = [
    "max(0.0, @Z.r)", "max(@Z.r, 0.0)", "min(max(0.0, @Z.r), 1.0)", "min(max(@Z.r, 0.0), 1.0)",
    "max(min(@Z.g, -0.0), -1.0)", "max(min(1.0, @Z.r), 0.0)",
]


@pytest.mark.parametrize("expr", _MINMAX_ROWS)
def test_minmax_literal_keeps_the_interpreters_signed_zero(expr):
    assert_parity(f"@OUT = vec3(atan2(0.0, {expr}));", {"Z": _signed_zeros()}, atol=0.0)


def test_nested_min_max_emits_its_operand_once():
    code = "@OUT = vec3(min(max(@A.r * 2.0 - 0.5, 0.0), 1.0), max(min(@A.g * 3.0, 0.8), 0.1), 0.0);"
    bindings = {"A": _img()}
    assert_parity(code, bindings, atol=0.0)
    bt = {n: infer_binding_type(v) for n, v in bindings.items()}
    program = Parser(Lexer(code).tokenize(), source=code).parse()
    program, tm, *_ = get_cache().compile_ast(program, bt, source=code)
    src = try_compile(program, tm)._tex_src
    assert "clamp" not in src and src.count("_torch.maximum(") == 2, src
    assert src.count("_bind['A'][..., 0]") == 1 and src.count("_bind['A'][..., 1]") == 1, src
