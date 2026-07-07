"""
v0.15.0 Phase 4 regression tests — defaults & QOL.
Q-4: fused-chain error attribution (stage-tagged SourceLoc).
CC-1: Triton-on-Windows hint (guidance only — smoke).
"""
from helpers import *
import TEX_Wrangle.tex_fusion as _FUS
from TEX_Wrangle.tex_compiler import ast_nodes as _A


def test_q4_stage_attribution(r: SubTestResult):
    print("\n--- Q-4: fused-chain error attribution ---")
    img = torch.rand(1, 4, 4, 4)
    stages = [
        {"code": "@OUT = @A * 0.5;", "chain_input": None, "bindings": {"A": img}},
        {"code": "float t = $amt; @OUT = @X + t;", "chain_input": "X", "bindings": {"amt": 0.1}},
    ]

    # Every fused node carries its originating stage index.
    try:
        prog, tm, ref, asg, par, used, merged = _FUS.compile_fused(stages, _infer_binding_type)
        seen = set()
        def walk(n):
            loc = getattr(n, "loc", None)
            if loc is not None and getattr(loc, "stage", None) is not None:
                seen.add(loc.stage)
            for c in _A.iter_child_nodes(n):
                walk(c)
        for s in prog.statements:
            walk(s)
        assert seen == {0, 1}, f"stage tags {sorted(seen)} != [0, 1]"
        r.ok("fused nodes tagged with originating stage (survives splice+optimize)")
    except Exception as e:
        r.fail("fused nodes tagged with originating stage", str(e))

    # SourceLoc gained the stage field without breaking the default construction.
    try:
        loc = _A.SourceLoc(3, 5)
        assert loc.stage is None and loc.line == 3 and loc.col == 5
        loc2 = _A.SourceLoc(1, 1, 2)
        assert loc2.stage == 2
        r.ok("SourceLoc.stage optional field (back-compatible)")
    except Exception as e:
        r.fail("SourceLoc.stage optional field", str(e))


def test_ct2_offset_sourceloc(r: SubTestResult):
    print("\n--- CT-2: offset-based lazy source locations ---")
    from TEX_Wrangle.tex_compiler.lexer import Lexer

    # Lazy offset resolves to the same 1-based line/col the eager form gave.
    try:
        src = "vec3 a = @A.rgb;\nfloat b = 2.0;\n  c = a * b;"
        # offset of the 'c' on line 3 (col 3, after two spaces)
        off = src.index("c = a")
        loc = _A.SourceLoc.from_offset(off, src)
        assert loc.line == 3 and loc.col == 3, f"got {loc.line}:{loc.col}, want 3:3"
        # start of file
        assert _A.SourceLoc.from_offset(0, src).col == 1
        # first char after a newline is col 1
        nl = src.index("\n") + 1
        assert _A.SourceLoc.from_offset(nl, src).col == 1
        r.ok("from_offset resolves correct 1-based line/col lazily")
    except Exception as e:
        r.fail("from_offset line/col resolution", str(e))

    # Token locations match what eager bookkeeping produced (multi-line source).
    try:
        src = "float x = 1.0;\nfloat yy = x + 2.0;\nvec3 c = vec3(yy);"
        toks = Lexer(src).tokenize()
        # find the 'yy' identifier token on line 2
        yy = next(t for t in toks if t.value == "yy")
        assert yy.loc.line == 2, f"yy on line {yy.loc.line}"
        assert yy.loc.col == src.split("\n")[1].index("yy") + 1
        r.ok("lexer token locations correct under offset-based loc")
    except Exception as e:
        r.fail("lexer token location correctness", str(e))

    # CT-2 and Q-4 coexist: an offset-backed loc still carries a mutable stage.
    try:
        loc = _A.SourceLoc.from_offset(5, "abcdefghij")
        loc.stage = 3
        assert loc.stage == 3 and loc.col == 6  # offset 5 → col 6 (no newline)
        r.ok("offset-backed loc carries mutable Q-4 stage")
    except Exception as e:
        r.fail("offset loc + Q-4 stage coexistence", str(e))
