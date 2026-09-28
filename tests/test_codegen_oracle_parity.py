"""Codegen emission bugs that served a different picture than the interpreter.

Invariant 2: the interpreter is the oracle and codegen matches it within 1e-5. Every row
here compiles through the engine pipeline a cook runs (type-check, optimize, re-check),
runs the interpreter and the codegen-only route on copies of the same bindings, and
requires the CODEGEN tier to have served the answer: a codegen crash that the route
quietly hands to the interpreter would otherwise compare the oracle against itself.

Each row's program is chosen so the optimizer cannot unroll or fold away the construct
under test (a `break`, a large trip count or a large body keeps a loop a loop).
"""
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
    ("empty range, step against the bound (scalar body)",
     "float acc = 0.0;\nfor (int i = 0; i < 10; i -= 1) { acc = acc + 1.0; if (acc > 500.0) { break; } }\n"
     "@OUT = vec3(acc) + @A.rgb * 0.0;", 0.0),
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
     "float acc = 0.0;\nfor (float x = 0.5; x < 2.5; x += 1.0) { acc = acc + 1.0; if (acc > 500.0) { break; } }\n"
     "@OUT = vec3(acc) + @A.rgb * 0.0;", 2.0),
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
