"""TRK-24 — a negative literal `$param` default (`f$k=-0.3;`) parses as
`UnaryOp('-', NumberLiteral)`, which `_check_param_decl`'s default-extraction chain
(`tex_compiler/type_checker.py`) had no branch for, so `default_value` silently stayed
`None`. Invisibility: `js/tex_extension.js`'s `PARAM_DECL_RE` (`:682`) already reads a
param's default from raw source text, independently of the type checker, so every
existing ComfyUI widget already showed the negative default — this fix only changes
what a bare `tex_api.compile/cook`, CLI, or other no-value caller sees.

Not wired into `run_both`/codegen — this is a compile-time metadata fold on the
declaration node, read only by `tex_engine.prepare`'s E6003-fallback and by hosts that
inspect `param_declarations` directly. Both tiers see the SAME `param_declarations`
dict (produced once, by the type checker, ahead of either tier), so there is no
interp/codegen divergence to test here (invariant #2 doesn't apply to this surface).
"""
from pathlib import Path

from helpers import *
from TEX_Wrangle import tex_engine
from TEX_Wrangle.tex_compiler.ast_nodes import UnaryOp, NumberLiteral

_EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

# The three shipped examples the tracker row names, and the default each declares —
# confirmed at base sha 6da837d by regex replay of the JS `PARAM_DECL_RE` against the
# source text (group 3 == "-0.3" / "-0.3" / "-0.5" exactly). `PARAM_DECL_RE`'s leading
# lookbehind is variable-width (`(?<=(?:^|[;{}])\s*)`), which Python's `re` cannot
# express, so these three are pinned explicitly rather than replayed live.
_SHIPPED_NEGATIVE_DEFAULTS = {
    "barrel_distortion.tex": ("k1", -0.3),
    "lens_distortion.tex": ("k1", -0.3),
    "recursive_pattern.tex": ("center_x", -0.5),
}


def test_negative_literal_defaults_recorded(r: SubTestResult):
    """A negative int or float literal default (any spacing the parser accepts) is
    recorded in `param_declarations`, not dropped to None."""
    print("\n--- TRK-24: negative literal param defaults ---")
    code = "f$a=-0.5; f$b=0.5; i$c=-3; f$d=- 0.25;\n@OUT = vec4($a, $b, float($c), $d);"
    try:
        _, checker = check_code(code)
        decls = checker.param_declarations
        assert decls["a"]["default_value"] == -0.5, decls["a"]
        assert decls["c"]["default_value"] == -3, decls["c"]
        assert isinstance(decls["c"]["default_value"], int), type(decls["c"]["default_value"])
        assert decls["d"]["default_value"] == -0.25, decls["d"]
        r.ok("negative literal defaults (int + float, incl. spaced unary minus) recorded")
    except Exception as e:
        r.fail("negative literal defaults recorded", f"{e}\n{traceback.format_exc()}")


def test_positive_defaults_unchanged(r: SubTestResult):
    """The new UnaryOp branch must not perturb the pre-existing NumberLiteral/
    StringLiteral/VecConstructor branches — positive and non-numeric defaults still
    come back exactly as before."""
    code = ('f$b=0.5; i$n=3; s$name="hi"; v3$col=vec3(1.0, 0.5, 0.0);\n'
            '@OUT = vec4($b, float($n), $col.x, $col.y);')
    try:
        _, checker = check_code(code)
        decls = checker.param_declarations
        assert decls["b"]["default_value"] == 0.5, decls["b"]
        assert decls["n"]["default_value"] == 3, decls["n"]
        assert isinstance(decls["n"]["default_value"], int), type(decls["n"]["default_value"])
        assert decls["name"]["default_value"] == "hi", decls["name"]
        assert decls["col"]["default_value"] == [1.0, 0.5, 0.0], decls["col"]
        r.ok("positive/string/vec defaults unchanged")
    except Exception as e:
        r.fail("positive defaults unchanged", f"{e}\n{traceback.format_exc()}")


def test_unary_op_ast_shape_matches_assumption(r: SubTestResult):
    """Pins the AST shape the fix depends on: `-0.3` really does parse as
    UnaryOp('-', NumberLiteral), not as a NumberLiteral with a negative value."""
    try:
        program = Parser(Lexer("f$k = -0.3;\n@OUT = vec4($k);").tokenize(),
                          source="f$k = -0.3;\n@OUT = vec4($k);").parse()
        param_decl = program.statements[0]
        assert isinstance(param_decl.default_expr, UnaryOp), type(param_decl.default_expr)
        assert param_decl.default_expr.op == "-", param_decl.default_expr.op
        assert isinstance(param_decl.default_expr.operand, NumberLiteral)
        r.ok("AST shape: negative literal default is UnaryOp('-', NumberLiteral)")
    except Exception as e:
        r.fail("AST shape assumption", f"{e}\n{traceback.format_exc()}")


def test_shipped_examples_declared_defaults(r: SubTestResult):
    """The three shipped examples the tracker names now declare the SAME negative
    default the JS widget builder's PARAM_DECL_RE already reads from source text —
    fixing the type checker changes no existing ComfyUI workflow's output."""
    print("\n--- TRK-24: shipped example defaults match the JS widget reading ---")
    for fname, (pname, expected) in _SHIPPED_NEGATIVE_DEFAULTS.items():
        try:
            path = _EXAMPLES / fname
            assert path.exists(), f"missing example: {path}"
            code = path.read_text(encoding="utf-8")
            _, checker = check_code(code)
            got = checker.param_declarations[pname]["default_value"]
            assert got == expected, f"{fname}: ${pname} default {got!r} != {expected!r}"
            r.ok(f"{fname}: ${pname} default == {expected} (matches JS PARAM_DECL_RE reading)")
        except Exception as e:
            r.fail(f"{fname}: declared default", f"{e}\n{traceback.format_exc()}")


def test_bare_cook_omitted_negative_param_uses_default(r: SubTestResult):
    """A bare `tex_engine.cook()` (what `tex_api.compile/cook` and CLI callers reach)
    that omits the negative-default param must use the declared default instead of
    raising E6003 — the bug the tracker filed as user-reachable."""
    print("\n--- TRK-24: bare cook with the negative param omitted ---")
    img = torch.rand(1, 4, 4, 3)
    cases = [
        ("barrel_distortion.tex", {"image": img}),
        ("lens_distortion.tex", {"image": img}),
        ("recursive_pattern.tex", {}),  # purely procedural — no @-bindings at all
    ]
    for fname, bindings in cases:
        try:
            code = (_EXAMPLES / fname).read_text(encoding="utf-8")
            res = tex_engine.cook(code, dict(bindings), device_mode="cpu")
            assert "OUT" in res.outputs, res.outputs.keys()
            assert torch.isfinite(res.outputs["OUT"]).all()
            r.ok(f"{fname}: bare cook with negative-default param omitted -> no E6003")
        except InterpreterError as e:
            r.fail(f"{fname}: bare cook omitted param", f"raised E6003-shaped error: {e}")
        except Exception as e:
            r.fail(f"{fname}: bare cook omitted param", f"{e}\n{traceback.format_exc()}")


def test_bare_cook_omitted_param_matches_explicit_default(r: SubTestResult):
    """Not just 'doesn't raise' — the value used is bit-exact to what an explicit
    binding of the same default would produce."""
    try:
        code = (_EXAMPLES / "barrel_distortion.tex").read_text(encoding="utf-8")
        img = torch.rand(1, 4, 4, 3)
        omitted = tex_engine.cook(code, {"image": img.clone()}, device_mode="cpu")
        explicit = tex_engine.cook(code, {"image": img.clone(), "k1": -0.3, "k2": 0.1},
                                   device_mode="cpu")
        assert torch.equal(omitted.outputs["OUT"], explicit.outputs["OUT"])
        r.ok("omitted-param cook bit-exact to the same default supplied explicitly")
    except Exception as e:
        r.fail("omitted param matches explicit default", f"{e}\n{traceback.format_exc()}")


def test_tex_node_widget_defaults_unchanged(r: SubTestResult):
    """`tex_node.py` owns no per-$param widget defaults at all (those are built by
    `js/tex_extension.js` from raw source text, independently of the type checker —
    the whole reason this fix is invisible). What it DOES own is the fixed static
    schema (code/device/compile_mode/precision/debug_nan_highlight); pin those so this
    type-checker-only change is confirmed not to have touched them."""
    print("\n--- TRK-24: tex_node.py static widget defaults unchanged ---")
    try:
        from TEX_Wrangle.tex_node import TEXWrangleNode as N
        schema = N.define_schema()
        by_id = {inp.id: inp for inp in schema.inputs}
        assert by_id["device"].default == "auto"
        assert by_id["compile_mode"].default == "none"
        assert by_id["precision"].default == "fp32"
        assert by_id["debug_nan_highlight"].default is False
        assert isinstance(by_id["code"].default, str) and "@OUT" in by_id["code"].default
        r.ok("tex_node.py static schema widget defaults unchanged")
    except Exception as e:
        r.fail("tex_node.py static widget defaults", f"{e}\n{traceback.format_exc()}")
