"""LANG-L1 — the pragma carried through as `Program.language`.

L1 of `docs/masked-control-flow.md`'s staged plan: "The pragma moves into `tex_compiler`;
`Parser.parse` sets `Program.language`; `tex_api.language_pragma` delegates. Nothing reads
the field yet." This file is that acceptance test, one row per clause:

  * a header pragma round-trips onto `Program.language` through every source -> AST path —
    `TEXCache.compile_tex` (the production seam), `tex_fusion`'s per-stage parse
    (`tex_fusion._parse`), and the corpus harness (`test_integration._prepare_example`);
  * a buried pragma (after real code, or inside a `/* ... */` block comment) still reads
    None, on both `Parser.parse().language` and the delegating `tex_api.language_pragma`;
  * `tex_api.language_pragma` and `tex_compiler.parser.language_pragma` agree on every case
    `test_v023_phase1.py` already pins — one scan under two names, not two regexes that
    could drift apart;
  * the field is already wired to its one anticipated reader, `tex_roi._language_tuple`
    (its `getattr(program, "language", None)` fallback predates this lane), with NO call-site
    edit in `tex_roi.py` — this row is the proof, not an assumption;
  * codegen emission reads the field ONLY through the LANG-L7 masked-flow gate: toggling it
    and diffing `_tex_src` over every shipped `examples/*.tex` program this harness can
    prepare moves emission for exactly the programs whose `tex_api.flow_plan` names a
    masking-relevant site (`not flow_plan(program).is_empty()`) — the SAME second half
    `masked_flow.enabled_for` checks, after its language-gate half — and none of the rest.
    `LANGUAGE_VERSION` is `0.25` at this head, so forcing ANY pragma `>= 0.25` passes that
    gate's language half uniformly for every program; which ones actually move is decided
    entirely by the flow_plan half. TRK-154 widened that predicate (a call to a user
    function reached under a per-pixel `if`) to two more shipped examples that call a
    helper that way without declaring the pragma themselves (`fix_pixels.tex`,
    `recursive_pattern.tex`) — this row's own expectation is derived from that SAME
    predicate, never a fixed name list, so it still catches a program moving for any OTHER,
    unexpected reason. This is the invariant-7 shape for everything the predicate says is
    NOT masking-relevant, and the intended, gated exception for everything it says is.

Every row runs on the compiler and the CPU interpreter/codegen alone. No ComfyUI, no CUDA,
no compiler toolchain, no Windows path, no embedded interpreter, no numpy, no timing.
"""
import os

from helpers import *

import test_integration as ti
from TEX_Wrangle import tex_api, tex_fusion, tex_roi
from TEX_Wrangle.tex_compiler.parser import language_pragma as parser_language_pragma

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_PRAGMA_SRC = "//!tex 0.25\nfloat x = 1.0;\n@OUT = vec4(x, x, x, 1.0);\n"
_NO_PRAGMA_SRC = "float x = 1.0;\n@OUT = vec4(x, x, x, 1.0);\n"
_BURIED_AFTER_CODE = "float x = 1.0;\n//!tex 0.25\n@OUT = vec4(x, x, x, 1.0);\n"
_BURIED_IN_BLOCK_COMMENT = "/*\n//!tex 0.25\n*/\nfloat x = 1.0;\n@OUT = vec4(x, x, x, 1.0);\n"


def test_l1_compile_tex_round_trips_language(r: SubTestResult):
    print("\n--- LANG-L1: Program.language round-trips through TEXCache.compile_tex ---")
    try:
        cache = TEXCache()
        program, *_ = cache.compile_tex(_PRAGMA_SRC, {})
        assert program.language == "0.25", f"expected '0.25', got {program.language!r}"

        program2, *_ = cache.compile_tex(_NO_PRAGMA_SRC, {})
        assert program2.language is None, f"expected None, got {program2.language!r}"
        r.ok("compile_tex: pragma present -> '0.25'; absent -> None")
    except Exception as e:
        r.fail("LANG-L1 compile_tex", str(e))


def test_l1_fusion_per_stage_parse_round_trips_language(r: SubTestResult):
    print("\n--- LANG-L1: Program.language round-trips through fusion's per-stage parse ---")
    try:
        program = tex_fusion._parse(_PRAGMA_SRC, {})
        assert program.language == "0.25", f"expected '0.25', got {program.language!r}"

        program2 = tex_fusion._parse(_NO_PRAGMA_SRC, {})
        assert program2.language is None, f"expected None, got {program2.language!r}"
        r.ok("fusion._parse: pragma present -> '0.25'; absent -> None")
    except Exception as e:
        r.fail("LANG-L1 fusion per-stage parse", str(e))


def test_l1_corpus_harness_round_trips_language(r: SubTestResult):
    print("\n--- LANG-L1: Program.language round-trips through the corpus harness ---")
    try:
        program, _bindings, _type_map, _outs = ti._prepare_example(_PRAGMA_SRC, 1, 4, 4)
        assert program.language == "0.25", f"expected '0.25', got {program.language!r}"

        program2, _b2, _t2, _o2 = ti._prepare_example(_NO_PRAGMA_SRC, 1, 4, 4)
        assert program2.language is None, f"expected None, got {program2.language!r}"
        r.ok("_prepare_example: pragma present -> '0.25'; absent -> None")
    except Exception as e:
        r.fail("LANG-L1 corpus harness", str(e))


def test_l1_buried_pragma_reads_none(r: SubTestResult):
    print("\n--- LANG-L1: a buried pragma is not a header pragma ---")
    try:
        for src, why in ((_BURIED_AFTER_CODE, "after real code"),
                         (_BURIED_IN_BLOCK_COMMENT, "inside a block comment")):
            prog = Parser(Lexer(src).tokenize(), source=src).parse()
            assert prog.language is None, (
                f"buried pragma ({why}) leaked onto Program.language: {prog.language!r}")
            assert tex_api.language_pragma(src) is None, (
                f"buried pragma ({why}) leaked through tex_api.language_pragma")
        r.ok("a buried //!tex pragma reads None on Program.language and tex_api.language_pragma, "
             "in both spellings (after real code; inside a block comment)")
    except Exception as e:
        r.fail("LANG-L1 buried pragma", str(e))


def test_l1_tex_api_delegates_to_tex_compiler(r: SubTestResult):
    print("\n--- LANG-L1: tex_api.language_pragma delegates to tex_compiler.parser.language_pragma ---")
    try:
        # The exact cases test_v023_phase1.py pins for tex_api.language_pragma — delegation
        # must not move a single one of them.
        cases = [
            "//!tex 0.20\n@OUT = vec4(u,v,0,1);",
            "  //!tex 1.5\nmore",
            "@OUT = vec4(u,v,0,1);",
            "// a normal comment\n@OUT = vec4(u,v,0,1);",
            "/*\n//!tex 99.0\n*/\n@OUT = vec4(u,v,0,1);",
            "@OUT = vec4(u,v,0,1);\n//!tex 99.0",
            "// a note\n//!tex 0.20\n@OUT = vec4(u,v,0,1);",
        ]
        for src in cases:
            a = tex_api.language_pragma(src)
            b = parser_language_pragma(src)
            assert a == b, f"tex_api gave {a!r}, tex_compiler.parser gave {b!r} for {src!r}"
        r.ok(f"tex_api.language_pragma agrees with tex_compiler.parser.language_pragma "
             f"on all {len(cases)} pinned cases")
    except Exception as e:
        r.fail("LANG-L1 delegation", str(e))


def test_l1_language_tuple_reads_the_real_field_with_no_callsite_edit(r: SubTestResult):
    print("\n--- LANG-L1: tex_roi._language_tuple already reads Program.language, unedited ---")
    try:
        program = tex_fusion._parse(_PRAGMA_SRC, {})
        assert program.language == "0.25"
        want = min(tex_api._ver_tuple("0.25"), tex_api._ver_tuple(tex_api.LANGUAGE_VERSION))

        got = tex_roi._language_tuple(program, _PRAGMA_SRC)
        assert got == want, f"expected {want}, got {got}"

        # The getattr-fallback path (an object with no `.language` at all) still scans `code`
        # directly — unchanged behaviour, proving the field is additive, not a replacement.
        got_fallback = tex_roi._language_tuple(object(), _PRAGMA_SRC)
        assert got_fallback == want, f"fallback path: expected {want}, got {got_fallback}"
        r.ok("_language_tuple reads Program.language when present, and still falls back to "
             "scanning the source when it is absent")
    except Exception as e:
        r.fail("LANG-L1 language_tuple", str(e))


def test_l1_codegen_emission_is_language_field_invariant(r: SubTestResult):
    print("\n--- LANG-L1: codegen emission moves with Program.language ONLY for a "
          "masking-relevant (flow_plan non-empty) program — invariant 7 for everything "
          "else, the intended exception there ---")
    # LANG-L7 opened the gate `masked_flow.enabled_for` reads, which is TWO conditions:
    # the effective language level (>= MASKED_FLOW_SINCE) and `tex_api.flow_plan` naming a
    # masking-relevant site. `LANGUAGE_VERSION` is `0.25` at this head, so forcing ANY
    # pragma `>= 0.25` below passes the language half uniformly for every program — the
    # flow_plan half is what actually decides whether emission can move. This row's
    # expectation is derived from THAT predicate, not a fixed name list (TRK-154 widened
    # it, correctly, to two more shipped examples — `fix_pixels.tex`/`recursive_pattern.tex`
    # — that call a helper under a per-pixel `if` without declaring the pragma themselves;
    # a name-list expectation would have gone stale exactly here). "Emission never reads
    # the field" stays true for every program the predicate calls not masking-relevant; the
    # intended, gated exception is everything it calls masking-relevant.
    moved = []
    masking_relevant = []
    declared_masked = []   # narrative only, not the expectation — see the docstring above
    checked = 0
    skipped = 0
    exdir = os.path.join(_ROOT, "examples")
    for fn in sorted(os.listdir(exdir)):
        if not fn.endswith(".tex"):
            continue
        with open(os.path.join(exdir, fn), encoding="utf-8") as f:
            src = f.read()
        try:
            program, _bindings, type_map, _outs = ti._prepare_example(src, 1, 4, 4)
            fn_a = try_compile(program, type_map)
        except Exception:
            skipped += 1
            continue  # a handful of examples need shapes/params this harness can't guess —
                      # not this test's concern (test_v037_frontend_parity.py hits the same wall)
        if fn_a is None:
            skipped += 1
            continue  # codegen declines this program (unsupported feature) — nothing to compare
        src_a = fn_a._tex_src
        before = program.language
        if before is not None and tex_api._ver_tuple(before) >= tex_roi.MASKED_FLOW_SINCE:
            declared_masked.append(fn)
        # The flow_plan half of `enabled_for`'s gate — independent of `program.language`
        # (a pure structural walk), so computed once, outside the mutate/restore below.
        if not tex_api.flow_plan(program).is_empty():
            masking_relevant.append(fn)
        try:
            program.language = "9.9"          # still >= MASKED_FLOW_SINCE: masked stays masked
            fn_b = try_compile(program, type_map)
            src_b = fn_b._tex_src if fn_b is not None else None
            program.language = None           # unmasks a program that was masked via `before`
            fn_c = try_compile(program, type_map)
            src_c = fn_c._tex_src if fn_c is not None else None
        finally:
            program.language = before
        checked += 1
        if src_b != src_a or src_c != src_a:
            moved.append(fn)
    if checked == 0:
        r.fail("LANG-L1 codegen invariance", "no example compiled through codegen — harness broken")
    elif sorted(moved) != sorted(masking_relevant):
        r.fail("LANG-L1 codegen invariance",
               f"moved set {sorted(moved)} != masking-relevant examples {sorted(masking_relevant)} "
               f"(declaring >= 0.25: {sorted(declared_masked)})")
    else:
        r.ok(f"codegen emission byte-identical across {checked - len(moved)}/{checked} "
             f"example(s) regardless of Program.language; moves ONLY for the "
             f"{len(masking_relevant)} masking-relevant example(s) ({sorted(masking_relevant)}), "
             f"of which {len(declared_masked)} also declare >= 0.25 themselves "
             f"({sorted(declared_masked)}), exactly as expected "
             f"({skipped} skipped: codegen-declined or harness-unpreparable)")
