"""DATA-6, the compiler half of "Planes": who owns the dot, and the swizzle splitback.

THE RULE. The lexer reads `@name.seg` as ONE binding token — `@ident` plus EXACTLY ONE dotted
segment when one immediately follows — so a plane is addressed by the name its file gives it
(`@beauty.diffuse`, `@beauty.Z`) and the sigil scan reports per-plane demand with no second
pass. One segment is the greed at which a plane read and a swizzle fall out of a single rule:
`@beauty.diffuse.rgb` is a plane read followed by an ordinary ChannelAccess, `@A.rgb` is a
swizzle the lexer cannot tell from a plane. The lexer never sees binding types, so the
splitback pass in `tex_cache.compile_ast` — ahead of the first TypeChecker — puts every dotted
binding that is NOT a plane read back to the `ChannelAccess(BindingRef(base), seg)` the parser
built before planes existed. Its last row is the compat guarantee in one line: an UNTYPED base
is a swizzle, so no program that compiled before planes can be re-read as a plane access.

PLANES is a WIRE-ONLY type, added exactly as ARRAY was: inert in every expression rule and
gated on the engine egress profile through the same switch, so under ComfyUI (the default) the
plane row never fires and every dotted `@` means what it always meant.

THE GREED IS OPT-IN (`Lexer(src, dotted_bindings=True)`), and only the production seam
(`TEXCache.compile_tex`) opts in. Every other tokenizer in the tree — the lazy-input analysis,
the ROI walk, the fused-chain splicer, the editor lint, the test harnesses — reads a binding's
name as the wire it is connected to, and keeps the pre-planes token stream byte for byte until
it is converged onto the seam deliberately. That default is pinned here; flipping it is a
decision, not a side effect.

Every row runs on the CPU interpreter or on the compiler alone. No ComfyUI, no CUDA, no
compiler toolchain, no Windows path, no embedded interpreter, no numpy — and no row asserts a
time. The one bit-exact comparison is same tier, same device, same bytes (see its marker).
"""
import os

from helpers import *

from TEX_Wrangle import tex_api, tex_lsp
from TEX_Wrangle.tex_cache import get_cache
from TEX_Wrangle.tex_compiler.ast_nodes import (
    Assignment, BindingRef, ChannelAccess, VarDecl,
)
from TEX_Wrangle.tex_compiler.lexer import BINDING_TYPE_PREFIXES
from TEX_Wrangle.tex_compiler.types import (
    TYPE_NAME_MAP, VALID_SWIZZLES, array_wires_enabled, set_array_wires,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _splitback(program, binding_types, *, source=""):
    """The pass under test, imported at call time (see the module docstring: red-first)."""
    from TEX_Wrangle.tex_cache import splitback_dotted_bindings
    return splitback_dotted_bindings(program, binding_types, source=source)


def _toks(src, **kw):
    kw.setdefault("dotted_bindings", True)
    return [(t.type, t.value) for t in Lexer(src, **kw).tokenize()[:-1]]   # drop EOF


def _parse(src):
    return Parser(Lexer(src, dotted_bindings=True).tokenize(), source=src).parse()


class _planes_enabled:
    """Flip the engine-profile switch for one block and restore it — the same flag
    `tex_marshalling.set_egress_profile("engine")` flips, read directly so this file
    touches no marshalling state."""
    def __init__(self, on=True):
        self._on = on

    def __enter__(self):
        self._prev = array_wires_enabled()
        set_array_wires(self._on)

    def __exit__(self, *a):
        set_array_wires(self._prev)


def _first_expr(src):
    """The initializer / value of the program's first statement."""
    stmt = _parse(src).statements[0]
    return stmt.initializer if isinstance(stmt, VarDecl) else stmt.value


# ── the lexer ───────────────────────────────────────────────────────────────

def test_dotted_at_binding_is_one_token(r: SubTestResult):
    print("\n--- DATA-6 L-B: `@beauty.diffuse` is ONE AT_BINDING token ---")
    try:
        toks = _toks("@beauty.diffuse")
        assert toks == [(TokenType.AT_BINDING, "beauty.diffuse")], toks
        r.ok("@beauty.diffuse -> one AT_BINDING, value verbatim")
    except Exception as e:
        r.fail("@beauty.diffuse -> one AT_BINDING, value verbatim", str(e))
    try:
        # the design's own example: the uppercase EXR data-layer name is a plane read on day one
        toks = _toks("@beauty.Z")
        assert toks == [(TokenType.AT_BINDING, "beauty.Z")], toks
        r.ok("@beauty.Z -> one token (uppercase Z is not in the collision set)")
    except Exception as e:
        r.fail("@beauty.Z -> one token", str(e))
    try:
        # the swizzle the lexer cannot tell from a plane: also one token, put back by the pass
        toks = _toks("@A.rgb")
        assert toks == [(TokenType.AT_BINDING, "A.rgb")], toks
        r.ok("@A.rgb -> one token (the splitback owns it)")
    except Exception as e:
        r.fail("@A.rgb -> one token", str(e))


def test_default_lexer_is_unchanged(r: SubTestResult):
    print("\n--- DATA-6 L-B: without the flag the token stream is the pre-planes one ---")
    cases = [
        ("@beauty.diffuse", [(TokenType.AT_BINDING, "beauty"), (TokenType.DOT, "."),
                             (TokenType.IDENT, "diffuse")]),
        ("@A.rgb", [(TokenType.AT_BINDING, "A"), (TokenType.DOT, "."), (TokenType.IDENT, "rgb")]),
        ("p@beauty.diffuse", [(TokenType.TYPED_AT_BINDING, "beauty"), (TokenType.DOT, "."),
                              (TokenType.IDENT, "diffuse")]),
        ("@a.b.c", [(TokenType.AT_BINDING, "a"), (TokenType.DOT, "."), (TokenType.IDENT, "b"),
                    (TokenType.DOT, "."), (TokenType.IDENT, "c")]),
    ]
    for src, want in cases:
        try:
            got = _toks(src, dotted_bindings=False)
            assert got == want, f"{src!r}: {got}"
            assert _toks(src, dotted_bindings=False) == \
                [(t.type, t.value) for t in Lexer(src).tokenize()[:-1]]    # the DEFAULT
            r.ok(f"default lexer: {src!r} -> pre-planes tokens")
        except Exception as e:
            r.fail(f"default lexer: {src!r}", str(e))
    try:
        assert Lexer("x").dotted_bindings is False
        r.ok("Lexer(...).dotted_bindings defaults to False")
    except Exception as e:
        r.fail("default is False", str(e))


def test_the_production_seam_lexes_greedily(r: SubTestResult):
    print("\n--- DATA-6 L-B: compile_tex is the seam that reads a dotted binding ---")
    src = "@OUT = @beauty.diffuse;"
    try:
        # With plane wires on and the base typed PLANES, a plane read survives compile_tex as
        # the dotted binding `beauty.diffuse` — only a greedy lexer can produce that name. (A
        # non-greedy lexer would build ChannelAccess(beauty, diffuse) and E3302 on `.diffuse`.)
        # Until the wire lane adds the per-plane rows, the read types by the VEC4 fallback.
        with _planes_enabled(True):
            program, type_map, referenced, *_ = get_cache().compile_tex(
                src, {"beauty": TEXType.PLANES})
        assert "beauty.diffuse" in referenced, referenced
        r.ok("compile_tex + planes on + PLANES base: `beauty.diffuse` is the referenced name")
    except Exception as e:
        r.fail("seam is greedy", str(e))
    try:
        # …and with plane wires OFF the same source through the same seam is the swizzle it
        # always was: `.diffuse` is not a swizzle pattern, E3302.
        try:
            get_cache().compile_tex(src + "// off\n", {"beauty": TEXType.PLANES})
            raise AssertionError("compiled")
        except TEXMultiError as e:                 # two swizzle errors accumulate (E3302 + E3303)
            codes = {d.code for d in e.diagnostics}
            assert "E3302" in codes, codes
        r.ok("compile_tex + planes off: the same read is a swizzle (E3302 on `.diffuse`)")
    except Exception as e:
        r.fail("seam splits back when off", str(e))


def test_one_segment_rule(r: SubTestResult):
    print("\n--- DATA-6 L-B: exactly ONE dotted segment, immediately adjacent ---")
    cases = [
        ("@a.b.c", [(TokenType.AT_BINDING, "a.b"), (TokenType.DOT, "."), (TokenType.IDENT, "c")]),
        ("@beauty.diffuse.rgb", [(TokenType.AT_BINDING, "beauty.diffuse"),
                                 (TokenType.DOT, "."), (TokenType.IDENT, "rgb")]),
        # not adjacent / not an identifier after the dot: the pre-planes token stream
        ("@A .r", [(TokenType.AT_BINDING, "A"), (TokenType.DOT, "."), (TokenType.IDENT, "r")]),
        ("@A..r", [(TokenType.AT_BINDING, "A"), (TokenType.DOT, "."), (TokenType.DOT, "."),
                   (TokenType.IDENT, "r")]),
        ("@A.5", [(TokenType.AT_BINDING, "A"), (TokenType.FLOAT_LIT, ".5")]),
        # `$` is never greedy — a dotted param stays two tokens (a swizzle of the param)
        ("$k.x", [(TokenType.DOLLAR_BINDING, "k"), (TokenType.DOT, "."), (TokenType.IDENT, "x")]),
        ("f$k.x", [(TokenType.TYPED_DOLLAR_BINDING, "k"), (TokenType.DOT, "."),
                   (TokenType.IDENT, "x")]),
        # a non-PLANES typed prefix declares a non-plane wire, so its dot is lexed as the swizzle
        ("v@A.rgb", [(TokenType.TYPED_AT_BINDING, "A"), (TokenType.DOT, "."),
                     (TokenType.IDENT, "rgb")]),
        ("@A.r.g", [(TokenType.AT_BINDING, "A.r"), (TokenType.DOT, "."), (TokenType.IDENT, "g")]),
    ]
    for src, want in cases:
        try:
            got = _toks(src)
            assert got == want, f"{src!r}: {got}"
            r.ok(f"one-segment rule: {src!r}")
        except Exception as e:
            r.fail(f"one-segment rule: {src!r}", str(e))
    try:
        # E1007 is unchanged: a sigil with no name after it
        for kw in ({}, {"dotted_bindings": True}):
            try:
                Lexer("@ + 1", **kw).tokenize()
                raise AssertionError("lexed")
            except LexerError:
                pass
        r.ok("E1007 (no name after the sigil) still raises, flag or no flag")
    except Exception as e:
        r.fail("E1007 still raises", str(e))


def test_p_prefix_declares_a_planes_wire(r: SubTestResult):
    print("\n--- DATA-6 L-B: the `p` typed-binding prefix ---")
    try:
        assert "p" in BINDING_TYPE_PREFIXES
        assert BINDING_HINT_TYPES["p"] is TEXType.PLANES
        r.ok("`p` is a binding prefix mapped to PLANES")
    except Exception as e:
        r.fail("`p` prefix row", str(e))
    try:
        toks = Lexer("p@beauty.diffuse", dotted_bindings=True).tokenize()[:-1]
        assert len(toks) == 1 and toks[0].type is TokenType.TYPED_AT_BINDING, toks
        assert toks[0].value == "beauty.diffuse" and toks[0].prefix == "p", toks
        r.ok("p@beauty.diffuse -> one TYPED_AT_BINDING (prefix p, value verbatim)")
    except Exception as e:
        r.fail("p@ is greedy", str(e))
    try:
        node = _first_expr("vec3 c = p@beauty.diffuse;")
        assert isinstance(node, BindingRef) and node.type_hint == "p" and node.kind == "wire"
        assert node.name == "beauty.diffuse"
        r.ok("the parser carries the hint on the dotted BindingRef")
    except Exception as e:
        r.fail("parser carries the p hint", str(e))


def test_collision_set_is_38_lowercase_names(r: SubTestResult):
    print("\n--- DATA-6 L-B: the collision set (design §1.2) ---")
    try:
        collision = set(CHANNEL_MAP) | set(VALID_SWIZZLES)
        assert len(CHANNEL_MAP) == 8, len(CHANNEL_MAP)
        assert len(VALID_SWIZZLES) == 30, len(VALID_SWIZZLES)
        assert len(collision) == 38, len(collision)
        r.ok("38 names = 8 channels + 30 swizzles")
    except Exception as e:
        r.fail("38 names", str(e))
    try:
        collision = set(CHANNEL_MAP) | set(VALID_SWIZZLES)
        assert all(n == n.lower() for n in collision), sorted(n for n in collision if n != n.lower())
        # the finding that matters most to a user: the conventional EXR data-layer names are
        # uppercase and collide with nothing — `@beauty.Z` is a plane read with no rename.
        for exr_name in ("Z", "N", "RGBA", "R", "G", "B", "A"):
            assert exr_name not in collision, exr_name
        r.ok("lowercase-only: Z / N / RGBA / R / G / B / A do not collide")
    except Exception as e:
        r.fail("lowercase-only", str(e))


def test_planes_is_inert_in_every_expression_rule(r: SubTestResult):
    print("\n--- DATA-6 L-B: PLANES is wire-only, inert, profile-gated (the ARRAY precedent) ---")
    try:
        t = TEXType.PLANES
        assert t.value == "planes"
        assert not t.is_vector and not t.is_scalar and not t.is_numeric
        assert not t.is_array and not t.is_string and not t.is_matrix
        assert t.is_planes and t.channels == 1 and t.mat_size == 0
        assert TYPE_NAME_MAP["planes"] is TEXType.PLANES
        r.ok("TEXType.PLANES: every predicate False, one TYPE_NAME_MAP row")
    except Exception as e:
        r.fail("TEXType.PLANES predicates", str(e))
    try:
        # the LSP's wire vocabulary IS TYPE_NAME_MAP, so a host can declare a PLANES wire
        got = tex_lsp._parse_binding_types({"beauty": "planes", "k": "float", "bad": "array"})
        assert got == {"beauty": TEXType.PLANES, "k": TEXType.FLOAT}, got
        r.ok("tex_lsp._parse_binding_types accepts `planes` (and still drops `array`)")
    except Exception as e:
        r.fail("LSP accepts planes", str(e))
    try:
        # the gate is the SAME switch as ARRAY's — a host cannot enable one without the other
        from TEX_Wrangle.tex_compiler.types import planes_wires_enabled
        with _planes_enabled(True):
            assert planes_wires_enabled() and array_wires_enabled()
        with _planes_enabled(False):
            assert not planes_wires_enabled() and not array_wires_enabled()
        assert planes_wires_enabled() == array_wires_enabled()
        r.ok("planes_wires_enabled follows array_wires_enabled (one switch)")
    except Exception as e:
        r.fail("one switch", str(e))
    try:
        # inert, ARRAY-parity: a PLANES-typed wire used AS A VALUE draws exactly the verdict an
        # ARRAY-typed one draws from the rules that key on the predicates — a typed declaration
        # refuses it (E3200), a constructor counts it as one component (E3601) — and the checker
        # never crashes (no E0000) on any shape, on either profile. (The two ARRAY-specific
        # guards, E3203 "assigning an array to @OUT" and E3300 channel access, are keyed on
        # `is_array` by name and do not yet have PLANES arms — that is a checker change owed
        # with the wire value, not a predicate.)
        shapes = ("@OUT = @beauty * 2.0;", "@OUT = vec4(@beauty, 1.0);",
                  "float f = float(@beauty); @OUT = vec4(f);", "@OUT = @beauty;",
                  "vec3 c = @beauty; @OUT = vec4(c, 1.0);", "@OUT = -@beauty;")
        for on in (False, True):
            with _planes_enabled(on):
                for src in shapes:
                    diags = tex_api.check(src, {"beauty": TEXType.PLANES})
                    assert all(d.code != "E0000" for d in diags), \
                        f"planes_on={on}: {src!r} crashed the checker: {[d.message for d in diags]}"
                codes = {d.code for d in tex_api.check(shapes[4], {"beauty": TEXType.PLANES})}
                assert "E3200" in codes, codes           # `vec3 c = @beauty` refused, as for ARRAY
                codes = {d.code for d in tex_api.check(shapes[1], {"beauty": TEXType.PLANES})}
                assert "E3601" in codes, codes           # one component, as for ARRAY
        r.ok("a PLANES value in an expression: ARRAY-parity verdicts, no crash, both profiles")
    except Exception as e:
        r.fail("PLANES inert in expressions", str(e))


# ── the splitback pass ──────────────────────────────────────────────────────

def _is_split(node, base, seg, hint=""):
    return (isinstance(node, ChannelAccess) and node.channels == seg
            and isinstance(node.object, BindingRef) and node.object.name == base
            and node.object.kind == "wire" and node.object.type_hint == hint)


def test_untyped_base_splits_back_to_a_swizzle(r: SubTestResult):
    print("\n--- DATA-6 L-B: an UNTYPED base is a swizzle (the compat guarantee) ---")
    src = "vec3 c = @image.rgb; float g = @image.g; @OUT = vec4(c, g);"
    try:
        prog = _parse(src)
        assert isinstance(prog.statements[0].initializer, BindingRef)            # greedy: dotted
        assert prog.statements[0].initializer.name == "image.rgb"
        _splitback(prog, {}, source=src)                                         # no types at all
        assert _is_split(prog.statements[0].initializer, "image", "rgb"), prog.statements[0]
        assert _is_split(prog.statements[1].initializer, "image", "g"), prog.statements[1]
        r.ok("binding_types={} : @image.rgb / @image.g -> ChannelAccess(BindingRef(image), ..)")
    except Exception as e:
        r.fail("untyped base splits back", str(e))
    try:
        # the same holds with plane wires ENABLED — untyped means swizzle on every profile
        with _planes_enabled(True):
            prog = _parse(src)
            _splitback(prog, {}, source=src)
            assert _is_split(prog.statements[0].initializer, "image", "rgb")
        r.ok("... and with plane wires enabled")
    except Exception as e:
        r.fail("untyped base splits back (planes on)", str(e))
    try:
        # through the production seam: referenced/assigned names are the BASES, the wire types
        program, type_map, referenced, assigned, params, used = \
            get_cache().compile_tex(src, {})
        assert "image" in referenced and not any("." in n for n in referenced), referenced
        assert set(assigned) == {"OUT"}, assigned
        r.ok("compile_tex: referenced holds `image`, never `image.rgb` / `image.g`")
    except Exception as e:
        r.fail("compile_tex referenced names are bases", str(e))
    try:
        # an assignment TARGET splits back too: `@OUT.rgb = ...` assigns OUT, not `OUT.rgb`
        src2 = "@OUT.rgb = @A.rgb; @OUT.a = 1.0;"
        prog = _parse(src2)
        assert isinstance(prog.statements[0], Assignment)
        assert isinstance(prog.statements[0].target, BindingRef)
        assert prog.statements[0].target.name == "OUT.rgb"
        _splitback(prog, {"A": TEXType.VEC4}, source=src2)
        assert _is_split(prog.statements[0].target, "OUT", "rgb")
        assert _is_split(prog.statements[1].target, "OUT", "a")
        r.ok("assignment targets: @OUT.rgb = ... -> ChannelAccess(BindingRef(OUT), rgb)")
    except Exception as e:
        r.fail("assignment target splits back", str(e))
    try:
        # a compound write desugars to a dotted target AND a dotted read; both split
        src3 = "@OUT.r += 0.5;"
        prog = _parse(src3)
        _splitback(prog, {}, source=src3)
        a = prog.statements[0]
        assert _is_split(a.target, "OUT", "r")
        assert _is_split(a.value.left, "OUT", "r"), a.value
        r.ok("`@OUT.r += ...` desugars to two dotted refs and both split back")
    except Exception as e:
        r.fail("compound assignment splits both sides", str(e))


def test_splitback_rows_and_their_mutations(r: SubTestResult):
    print("\n--- DATA-6 L-B: every row of the splitback table, mutated both ways ---")
    src = "vec3 c = @beauty.diffuse;"

    def run(bt, on, hint_src=None):
        s = hint_src or src
        with _planes_enabled(on):
            prog = _parse(s)
            _splitback(prog, bt, source=s)
        return prog.statements[0].initializer

    rows = [
        # (label, binding_types, planes_on, expect_kept)
        ("PLANES base, planes ON  -> kept (a plane read)", {"beauty": TEXType.PLANES}, True, True),
        ("PLANES base, planes OFF -> split (ComfyUI never sees planes)",
         {"beauty": TEXType.PLANES}, False, False),
        ("expanded plane row, planes ON -> kept", {"beauty.diffuse": TEXType.VEC3}, True, True),
        ("expanded plane row, planes OFF -> split", {"beauty.diffuse": TEXType.VEC3}, False, False),
        ("VEC4 base -> split", {"beauty": TEXType.VEC4}, True, False),
        ("VEC3 base -> split", {"beauty": TEXType.VEC3}, True, False),
        ("FLOAT (mask) base -> split (existing rules own `.r` there)",
         {"beauty": TEXType.FLOAT}, True, False),
        ("STRING base -> split (E3300 fires later, unchanged)", {"beauty": TEXType.STRING}, True, False),
        ("ARRAY base -> split", {"beauty": TEXType.ARRAY}, True, False),
        ("absent base -> split", {}, True, False),
        ("absent base, planes OFF -> split", {}, False, False),
        ("another wire typed PLANES does not make THIS base one -> split",
         {"other": TEXType.PLANES}, True, False),
    ]
    for label, bt, on, kept in rows:
        try:
            node = run(bt, on)
            if kept:
                assert isinstance(node, BindingRef) and node.name == "beauty.diffuse", node
            else:
                assert _is_split(node, "beauty", "diffuse"), node
            r.ok(f"row: {label}")
        except Exception as e:
            r.fail(f"row: {label}", str(e))
    # the `p@` hint is a type source for the base, exactly as the checker's hint rule is
    try:
        node = run({}, True, "vec3 c = p@beauty.diffuse;")
        assert isinstance(node, BindingRef) and node.name == "beauty.diffuse" \
            and node.type_hint == "p", node
        r.ok("row: `p@beauty.diffuse`, planes ON, no map -> kept (the hint types the base)")
    except Exception as e:
        r.fail("row: p@ hint keeps", str(e))
    try:
        node = run({}, False, "vec3 c = p@beauty.diffuse;")
        assert _is_split(node, "beauty", "diffuse", hint="p"), node
        r.ok("row: `p@beauty.diffuse`, planes OFF -> split, hint kept on the base")
    except Exception as e:
        r.fail("row: p@ hint splits when off", str(e))
    # the split is on the LAST dot, and a kept plane can be swizzled for free
    try:
        s = "vec3 c = @beauty.diffuse.rgb;"
        with _planes_enabled(True):
            prog = _parse(s)
            _splitback(prog, {"beauty": TEXType.PLANES}, source=s)
        node = prog.statements[0].initializer
        assert isinstance(node, ChannelAccess) and node.channels == "rgb"
        assert isinstance(node.object, BindingRef) and node.object.name == "beauty.diffuse"
        r.ok("`@beauty.diffuse.rgb` (planes ON) -> ChannelAccess(plane read, rgb)")
    except Exception as e:
        r.fail("swizzled plane read", str(e))
    try:
        s = "vec3 c = @beauty.diffuse.rgb;"
        with _planes_enabled(True):
            prog = _parse(s)
            _splitback(prog, {"beauty": TEXType.VEC4}, source=s)
        node = prog.statements[0].initializer
        assert isinstance(node, ChannelAccess) and node.channels == "rgb"
        assert _is_split(node.object, "beauty", "diffuse"), node
        r.ok("`@beauty.diffuse.rgb` (VEC4 base) -> ChannelAccess(ChannelAccess(beauty, diffuse), rgb)")
    except Exception as e:
        r.fail("double swizzle on a vector base", str(e))
    # the diagnostic column is the SEGMENT's, as the parser's ChannelAccess always carried
    try:
        s = "@OUT = @A.q;"
        try:
            get_cache().compile_tex(s, {"A": TEXType.VEC4})
            raise AssertionError("compiled")
        except TypeCheckError as e:
            assert e._code == "E3301", e._code
            assert (e.loc.line, e.loc.col) == (1, 11), (e.loc.line, e.loc.col)
        r.ok("E3301 on `@A.q` lands on column 11 -- the segment, as before")
    except Exception as e:
        r.fail("diagnostic column preserved", str(e))
    # the cache key already separates the two readings: same source, different binding types
    try:
        fp_planes = TEXCache.fingerprint(src, {"beauty": TEXType.PLANES})
        fp_vec = TEXCache.fingerprint(src, {"beauty": TEXType.VEC4})
        assert fp_planes != fp_vec
        r.ok("fingerprint(code, {beauty: PLANES}) != fingerprint(code, {beauty: VEC4})")
    except Exception as e:
        r.fail("fingerprint separates the readings", str(e))


def test_splitback_is_an_identity_on_the_cook(r: SubTestResult):
    print("\n--- DATA-6 L-B: the split-back program cooks the SAME pixels ---")
    # `@image.rgb` (compile_tex lexes greedily -> one token -> split back) versus
    # `@image .rgb` (a space: no lexer fuses it, so the parser builds the ChannelAccess
    # itself). Both go through compile_tex and the CPU interpreter: same tier, same device,
    # same bindings, so the outputs must be bit-identical -- the splitback is a rewrite to
    # the SAME AST.
    greedy = ("vec3 c = @image.rgb * 0.5 + vec3(@image.r, @image.g, @image.b) * 0.25;\n"
              "@OUT = vec4(c, @image.a);\n")
    spaced = greedy.replace("@image.", "@image .")
    A = make_img(2, 8, 8, 4, seed=7)
    bt = {"image": TEXType.VEC4}
    outs = []
    try:
        for code in (greedy, spaced):
            program, type_map, referenced, assigned, params, used = get_cache().compile_tex(code, bt)
            assert "image" in referenced and not any("." in n for n in referenced), referenced
            res = Interpreter().execute(program, {"image": A}, type_map, device="cpu",
                                        output_names=["OUT"], source=code)
            outs.append(res["OUT"])
        # lnt2-ok: same tier (CPU interpreter, no dispatcher), same device, same bytes in -- the splitback must be an identity
        assert torch.equal(outs[0], outs[1]), (outs[0] - outs[1]).abs().max().item()
        r.ok("`@image.rgb` and `@image .rgb` cook bit-identically on the CPU interpreter")
    except Exception as e:
        r.fail("splitback is an identity on the cook", str(e))
    try:
        # ...and it survives the engine profile being ON: an untyped/vector base is still a swizzle
        with _planes_enabled(True):
            program, type_map, *_ = get_cache().compile_tex(greedy + "// planes-on\n", bt)
            res = Interpreter().execute(program, {"image": A}, type_map, device="cpu",
                                        output_names=["OUT"], source=greedy)
        # lnt2-ok: same tier (CPU interpreter, no dispatcher), same device, same bytes in -- the profile switch must not move a swizzle
        assert torch.equal(res["OUT"], outs[0])
        r.ok("... and with plane wires enabled")
    except Exception as e:
        r.fail("identity with planes on", str(e))


def test_swizzle_sugar_stays_refused(r: SubTestResult):
    print("\n--- DATA-6 L-B: `@A.rgb[..]` / `@A.rgb(..)` stay the compile errors they were ---")
    cases = [
        ("@OUT = @A.rgb(u, v);", "E2002"),        # was E2002 from the parser (ChannelAccess is not callable)
        ("@OUT = @A.rgb[ix, iy];", "E2000"),      # was a parse error (E2000) on the `,`
    ]
    for src, code in cases:
        try:
            try:
                get_cache().compile_tex(src, {"A": TEXType.VEC4})
                raise AssertionError("compiled")
            except TypeCheckError as e:
                assert e._code == code, f"{src!r}: got {e._code}, want {code}"
            r.ok(f"{src!r} -> {code}")
        except Exception as e:
            r.fail(f"{src!r} -> {code}", str(e))
    try:
        # and the sugar on a bare binding, then a swizzle, is unchanged and compiles
        program, type_map, referenced, *_ = get_cache().compile_tex(
            "@OUT = vec4(@A[ix, iy].rgb, @A(u, v).a);", {"A": TEXType.VEC4})
        assert "A" in referenced and not any("." in n for n in referenced), referenced
        r.ok("`@A[ix, iy].rgb` / `@A(u, v).a` still compile")
    except Exception as e:
        r.fail("bare-binding sugar then swizzle", str(e))


def test_tripwire_is_portable(r: SubTestResult):
    print("\n--- DATA-6 L-B: tools/planes_compat_scan.py derives its root from __file__ ---")
    try:
        p = os.path.join(_ROOT, "tools", "planes_compat_scan.py")
        with open(p, encoding="utf-8") as f:
            src = f.read()
        assert "G:\\" not in src and "ComfyUI_Menu" not in src, "a box-specific path survives"
        assert "__file__" in src
        r.ok("no drive-letter path; root derived from __file__")
    except Exception as e:
        r.fail("tripwire portable", str(e))
