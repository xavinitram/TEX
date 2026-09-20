"""PERF-4 — the two remaining front-end re-scans, removed without moving an answer.

TWO CHANGES, ONE SHAPE. Both are "the analysis re-reads what it has already read".

  F2  `tex_lazy.lazy_required_bindings` keyed its answer memo on the parameter VALUES (which
      is right — folding a `$param` to a literal is the whole of tier T3) and then called
      `parse_and_split` on EVERY miss, so a moving widget paid a full `Lexer.tokenize` +
      `Parser.parse` for a source that had not changed. The parse is now memoized per source
      (`tex_lazy._pristine_program`) and each evaluation folds its own `ast_nodes.clone_tree`
      copy — the fix `tex_roi` took in PERF-1, on the module PERF-1's finding F2 named.

  F4  `tex_roi._has_ungrounded_halo` traversed the program twice (a recursive `_scan` for
      case (1), then a stack walk for case (2)) and inside the second walk re-descended each
      initializer/value through `_subtree_has_halo`, so a halo call deep in an expression was
      visited once per ancestor statement. It is now ONE traversal that returns "this subtree
      contains a halo call" on the way back up.

WHY THIS FILE IS SHAPED LIKE A DERIVATION ORACLE (PERF-1's shape, deliberately). Both changes
are behaviour-preserving or they are nothing, and "behaviour" is an answer over an open set of
sources and an open set of parameter valuations. So the PRE-CHANGE implementations are kept
below — captured from base sha `c9dde51`, self-contained so that patching a `tex_roi` helper
mutates the SHIPPED side only — and the two are run against each other over every shipped
`examples/*.tex`, the ten-stage host-demo comp, and a hand-written corpus of the shapes where
a value (F2) or a name boundary (F4) is known to move the answer. `_ORACLE_SENSITIVE_ROWS`
and `test_perf4_halo_corpus_is_not_vacuous` pin that the corpora really contain such shapes,
so neither comparison can pass by being empty.

BOTH DIRECTIONS. `test_perf4_the_lazy_clone_is_load_bearing` hands the fold the memoized parse
itself and requires the oracle to NOTICE; `test_perf4_halo_mutants_are_caught` breaks the
merged scan's two load-bearing rules (the bottom-up "does this subtree carry a halo", and
case (1)'s grounded/ungrounded distinction) and requires the corpus to notice each. The
red-first halves are `test_perf4_a_lazy_source_is_parsed_once` (N values cost N lexes on the
base sha) and `test_perf4_the_halo_scan_visits_each_node_once` (every node is descended at
least twice on the base sha).

WHERE THE LAZY ANALYSIS IS CALLED FROM, measured rather than assumed (it decides what these
counts mean): `tex_node.check_lazy_status` (ComfyUI's lazy-input round, re-invoked as wired
scalars arrive), `tex_engine.prepare`'s E6003 gate when the caller passes `forgive_dead_refs`
(only `tex_node` does), and `tex_fusion.fused_required_bindings` (built, deliberately not
wired). `benchmarks/host_path_counts.py`'s seven scenarios reach NONE of them — they drive
`tex_api`/`tex_engine` directly, where `forgive_dead_refs` is off by default — so F2 moves no
`tests/test_bench2_counts.py` row and is gated here instead.

PORTABILITY: CPU, no ComfyUI, no CUDA, no compiler, no numpy, no timing assertion.
"""
import glob
import os

from helpers import *

from TEX_Wrangle import tex_lazy, tex_roi
from TEX_Wrangle.tex_cache import parse_and_split
from TEX_Wrangle.tex_compiler.ast_nodes import (
    Assignment, ForLoop, FunctionCall, FunctionDef, NumberLiteral, VarDecl, WhileLoop,
    iter_child_nodes,
)
from TEX_Wrangle.tex_compiler.optimizer import _fold_all, _propagate_literal_locals
from TEX_Wrangle.tex_lazy import _fp32, _substitute_params
from TEX_Wrangle.tex_marshalling import sigil_names

# The isolation every oracle row in BOTH front-end files needs, defined once beside PERF-1's
# rows and imported here (the suite already imports one test module from another —
# `tests/compat_corpus.py` takes `test_integration._prepare_example`). Its docstring carries
# the whole diagnosis: the analysis parse memos key a whole AST on the source text while the
# tree they hold depends on the process-global plane-wire flag, so an earlier file that parsed
# a dotted example with plane wires ON makes `test_perf4_lazy_answers_are_identical` below
# report `examples/aov_relight.tex`'s plane reads as moved answers.
from test_perf1_roi_walk_memo import isolated_analysis

# Bound HERE, at import, so the oracles below keep calling the real helpers while a test
# monkeypatches `tex_roi.<name>` to mutate the shipped side. An oracle that read the module
# attribute at call time would move with the mutation and prove nothing.
_footmap = tex_roi._footmap
_write_target_name = tex_roi._write_target_name
_collect_read_names = tex_roi._collect_read_names

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── the pre-change implementations, kept as the oracles (base sha c9dde51) ───

def _base_is_halo_call(n, fm) -> bool:
    """`tex_roi._is_halo_call`, copied rather than imported: the shipped one is inside the
    thing under test, and a mutation test patches it."""
    if n.__class__ is not FunctionCall:
        return False
    fp = fm.get(n.name)
    return isinstance(fp, tuple) and len(fp) >= 1 and fp[0] in ("halo", "halo_arg")


def _base_subtree_has_halo(node, fm) -> bool:
    """`tex_roi._subtree_has_halo` as it stood at the base sha — the re-descent F4 removes."""
    stack = [node]
    while stack:
        n = stack.pop()
        if _base_is_halo_call(n, fm):
            return True
        stack.extend(iter_child_nodes(n))
    return False


def _base_has_ungrounded_halo(program) -> bool:
    """`tex_roi._has_ungrounded_halo` at the base sha: two traversals plus the re-descent."""
    fm = _footmap()

    def _scan(node, ungrounded: bool) -> bool:
        if ungrounded and _base_is_halo_call(node, fm):
            return True
        cls = node.__class__
        if cls is VarDecl:
            return node.initializer is not None and _scan(node.initializer, True)
        if cls in (FunctionDef, ForLoop, WhileLoop):
            return any(_scan(ch, True) for ch in iter_child_nodes(node))
        return any(_scan(ch, ungrounded) for ch in iter_child_nodes(node))

    if any(_scan(s, False) for s in program.statements):
        return True
    halo_named = set()
    stack = list(program.statements)
    while stack:
        n = stack.pop()
        cls = n.__class__
        if cls is VarDecl and n.initializer is not None and \
                _base_subtree_has_halo(n.initializer, fm):
            halo_named.add(n.name)
        elif cls is Assignment and _base_subtree_has_halo(n.value, fm):
            tn = _write_target_name(n.target)
            if tn is None:
                return True
            halo_named.add(tn)
        stack.extend(iter_child_nodes(n))
    if not halo_named:
        return False
    reads = set()
    for s in program.statements:
        _collect_read_names(s, reads)
    return bool(halo_named & reads)


def _mutant_has_ungrounded_halo(program, *, shallow_value=False, ignore_ungrounded=False):
    """The MERGED scan with one of its two load-bearing rules removed, for the mutation rows.

    `shallow_value` asks only whether the assigned VALUE NODE is itself a halo call instead of
    whether its subtree carries one — i.e. drops the bottom-up propagation that replaced
    `_subtree_has_halo`. `ignore_ungrounded` lets a GROUNDED halo call answer case (1), which
    is the over-blocking direction. Each must be caught by the corpus below or the corpus is
    not exercising what the merge rests on."""
    fm = _footmap()
    state = {"case1": False, "unnameable": False}
    halo_named: set = set()

    def _visit(node, ungrounded: bool, scanned: bool) -> bool:
        cls = node.__class__
        has = cls is FunctionCall and _base_is_halo_call(node, fm)
        if has and (ignore_ungrounded or ungrounded) and scanned:
            state["case1"] = True
        if cls is VarDecl:
            init, init_has = node.initializer, False
            for ch in iter_child_nodes(node):
                if init is not None and ch is init:
                    if _visit(ch, True, scanned):
                        init_has = has = True
                elif _visit(ch, ungrounded, False):
                    has = True
            if init_has and not (shallow_value and init.__class__ is not FunctionCall):
                halo_named.add(node.name)
            return has
        inner = True if cls in (FunctionDef, ForLoop, WhileLoop) else ungrounded
        if cls is Assignment:
            value_has = False
            for ch in iter_child_nodes(node):
                if _visit(ch, inner, scanned):
                    has = True
                    if ch is node.value:
                        value_has = True
            if shallow_value:
                value_has = _base_is_halo_call(node.value, fm)
            if value_has:
                tn = _write_target_name(node.target)
                if tn is None:
                    state["unnameable"] = True
                else:
                    halo_named.add(tn)
            return has
        for ch in iter_child_nodes(node):
            if _visit(ch, inner, scanned):
                has = True
        return has

    for s in program.statements:
        _visit(s, False, True)
    if state["case1"] or state["unnameable"]:
        return True
    if not halo_named:
        return False
    reads = set()
    for s in program.statements:
        _collect_read_names(s, reads)
    return bool(halo_named & reads)


def _base_lazy_required_bindings(code: str, param_values: dict | None = None):
    """`tex_lazy.lazy_required_bindings` at the base sha: UNMEMOIZED, a FRESH parse per call.

    Everything after the parse is the shipped code — this file imports the same
    `_substitute_params` / `_fold_all` / `_propagate_literal_locals` / `_prune_static_flow` /
    `_collect_binding_refs` the module uses, so the oracle differs from the implementation in
    exactly the thing under test and drifts with the analysis if the analysis ever changes."""
    param_values = param_values or {}
    try:
        program = parse_and_split(code, {})
        subs = {
            name: NumberLiteral(value=_fp32(v), is_int=isinstance(v, (bool, int)))
            for name, v in param_values.items()
            if isinstance(v, (bool, int, float))
        }
        stmts = program.statements
        if subs:
            for stmt in stmts:
                _substitute_params(stmt, subs)
        stmts = _fold_all(stmts)
        stmts = _propagate_literal_locals(stmts)
        stmts = _fold_all(stmts)
        stmts = tex_lazy._prune_static_flow(stmts)
        return tex_lazy._collect_binding_refs(stmts)
    except Exception:
        return None


# ── the corpora ──────────────────────────────────────────────────────────────

#: Sources whose LAZY answer is known to move with a value (tier T3), so the F2 comparison
#: cannot pass for an analysis that ignored its parameters.
_LAZY_SENSITIVE = {
    "gated_input": "@OUT = ($k > 0.5) ? @A : @B;",
    "gated_block": "if ($k > 0.5) { @OUT = @A; } else { @OUT = @B * @C; }",
    "gate_via_local": "float g = $k * 2.0;\n@OUT = (g > 1.0) ? @A : @B;",
    "never_severed": "@OUT = @A * 0.0 + @B * $k;",     # invariant 11: `*0` must not sever @A
}

#: Sources whose HALO answer turns on a rule the merged scan carries. Named, not discovered.
_HALO_SHAPES = {
    "grounded": "@OUT = gauss_blur(@A, 2.0);",
    "case1_vardecl": "vec4 b = gauss_blur(@A, 2.0);\n@OUT = gauss_blur(b, 2.0);",
    "case1_loop": "@OUT = @A;\nfor (int i = 0; i < 2; i = i + 1) { @OUT = gauss_blur(@A, 2.0); }",
    "case2_direct": "@T = gauss_blur(@A, 2.0);\n@OUT = @T * 2.0;",
    "case2_nested": "@T = vec4(gauss_blur(@A, 2.0).rgb, 1.0);\n@OUT = @T * 2.0;",
    "case2_in_if": ("if ($k > 0.5) { @T = gauss_blur(@A, 2.0); @OUT = gauss_blur(@T, 2.0); }"
                    " else { @OUT = @A; }"),
    "named_no_halo": "@T = @A * 2.0;\n@OUT = gauss_blur(@T, 2.0);",
    "written_unread": "@T = gauss_blur(@A, 2.0);\n@OUT = @B * 2.0;",
}

#: Shared with PERF-1's file by intent, not by import: the valuations each corpus source is
#: analysed under. Each is a shape that has moved an answer here before.
_VALUATIONS = (0.0, 1.0, 0.5, 2.0, 2, True, -1.0, 3.0, 7.0, 0.25, float("nan"),
               float("inf"), 1e-8)


def _corpus():
    """`(label, code)` for every shipped example, the host-demo comp and both shape sets."""
    rows = [(f"lazy:{k}", v) for k, v in _LAZY_SENSITIVE.items()]
    rows += [(f"halo:{k}", v) for k, v in _HALO_SHAPES.items()]
    demo = os.path.join(_ROOT, "examples", "host_demo.py")
    if os.path.exists(demo):
        import importlib.util
        spec = importlib.util.spec_from_file_location("_perf4_host_demo", demo)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        rows += [(f"comp:{n}", c) for n, c, _p in mod._COMP_STAGES]
    for path in sorted(glob.glob(os.path.join(_ROOT, "examples", "*.tex"))):
        with open(path, "r", encoding="utf-8") as fh:
            rows.append(("example:" + os.path.basename(path), fh.read()))
    return rows


def _valuations_for(code: str):
    """Every `$name` in the source bound to each value of `_VALUATIONS`, plus the empty dict.

    The empty dict is its own program shape: `tex_roi._fold_program` runs the optimizer's fold
    only when at least one foldable param is supplied."""
    names = sorted(sigil_names(code)[1])
    out = [{}]
    for v in _VALUATIONS:
        out.append({n: v for n in names})
    return out


def _folded(code: str, params: dict):
    """The folded program `_walk` hands `_has_ungrounded_halo`, or None if it will not parse.

    A FRESH fold per side of a comparison: the analysis does not mutate its input, but a
    shared tree would make a mutation test's failure look like a shared-state bug."""
    try:
        return tex_roi._fold_program(code, params)
    except Exception:
        return None


# ── F2: the lazy analysis ────────────────────────────────────────────────────

@isolated_analysis
def test_perf4_lazy_answers_are_identical(r: SubTestResult):
    """Every `(source, valuation)` answers what the pre-change analysis answered."""
    print("\n--- PERF-4 F2: the memoized-parse lazy analysis vs the pre-change one ---")
    rows = _corpus()
    bad, checked = [], 0
    for label, code in rows:
        for params in _valuations_for(code):
            got = tex_lazy.lazy_required_bindings(code, params)
            want = _base_lazy_required_bindings(code, params)
            checked += 1
            if got != want:
                bad.append(f"{label} params={params}: {want!r} -> {got!r}")
    if bad:
        r.fail("PERF-4 lazy identity",
               f"{len(bad)} of {checked} answers moved; first 3: {bad[:3]}")
    else:
        r.ok(f"{checked} lazy answers over {len(rows)} sources are unchanged")


@isolated_analysis
def test_perf4_a_lazy_source_is_parsed_once(r: SubTestResult):
    """RED-FIRST. Many parameter values over one source cost ONE lex and ONE parse.

    Counted through the real front end rather than through `_pristine_program`, so a fix that
    moved the parse somewhere else would not satisfy it. On the base sha this reads 24/24."""
    print("\n--- PERF-4 F2: one lex + one parse per distinct source ---")
    from TEX_Wrangle.tex_compiler import lexer as _lx, parser as _ps
    code = "@OUT = ($k > 0.5) ? @A : @B * $gain;"
    tex_lazy.clear_lazy_memo()
    n = {"lex": 0, "parse": 0}
    lex0, parse0 = _lx.Lexer.tokenize, _ps.Parser.parse

    def lex(self, *a, **k):
        n["lex"] += 1
        return lex0(self, *a, **k)

    def parse(self, *a, **k):
        n["parse"] += 1
        return parse0(self, *a, **k)

    _lx.Lexer.tokenize, _ps.Parser.parse = lex, parse
    try:
        for i in range(24):
            tex_lazy.lazy_required_bindings(code, {"k": 0.01 * i, "gain": 1.0 + i})
    finally:
        _lx.Lexer.tokenize, _ps.Parser.parse = lex0, parse0
    if (n["lex"], n["parse"]) != (1, 1):
        r.fail("PERF-4 lazy parse-once",
               f"24 parameter values cost {n['lex']} lex(es) and {n['parse']} parse(s); "
               f"expected 1 and 1")
    else:
        r.ok("24 parameter values over one source cost 1 lex and 1 parse")

    # ... and a SECOND source is still parsed, so the memo is a cache and not a swallow.
    n["lex"] = n["parse"] = 0
    other = "@OUT = @IN * $lift;"
    _lx.Lexer.tokenize, _ps.Parser.parse = lex, parse
    try:
        tex_lazy.lazy_required_bindings(other, {"lift": 0.1})
    finally:
        _lx.Lexer.tokenize, _ps.Parser.parse = lex0, parse0
    r.ok("a second source is parsed once more (the memo keys on the source)") \
        if (n["lex"], n["parse"]) == (1, 1) else \
        r.fail("PERF-4 lazy parse-once",
               f"a NEW source cost {n['lex']}/{n['parse']} lex/parse, expected 1/1 — the memo "
               f"is answering for the wrong source")


@isolated_analysis
def test_perf4_the_lazy_clone_is_load_bearing(r: SubTestResult):
    """MUTATION. Hand the analysis the memoized parse ITSELF and the oracle must notice.

    The memo holds a PRISTINE parse and the substitution rewrites in place, so without the
    copy the second value folds a tree the first value already folded — literals from the
    previous tick, permanently."""
    print("\n--- PERF-4 F2 mutation: the analysis's copy is load-bearing ---")
    code = _LAZY_SENSITIVE["gated_input"]
    vals = (0.0, 1.0, 0.5)
    want = [_base_lazy_required_bindings(code, {"k": v}) for v in vals]
    real = tex_lazy.clone_tree
    tex_lazy.clear_lazy_memo()
    tex_lazy.clone_tree = lambda p: p            # the broken version: no copy at all
    try:
        got = [tex_lazy.lazy_required_bindings(code, {"k": v}) for v in vals]
    finally:
        tex_lazy.clone_tree = real
        tex_lazy.clear_lazy_memo()
    r.ok("an analysis without the copy is CAUGHT by the oracle (the copy is load-bearing)") \
        if got != want else \
        r.fail("PERF-4 lazy mutation",
               "removing the AST copy changed no answer — the oracle is not sensitive to the "
               "thing this change rests on")

    tex_lazy.clear_lazy_memo()
    fixed = [tex_lazy.lazy_required_bindings(code, {"k": v}) for v in vals]
    r.ok("with the copy restored the same three valuations agree with the oracle") \
        if fixed == want else \
        r.fail("PERF-4 lazy mutation", f"restored analysis still disagrees: {want!r} -> {fixed!r}")


@isolated_analysis
def test_perf4_the_lazy_memo_hands_out_no_shared_ast(r: SubTestResult):
    """The memoized parse is never handed to a caller, and survives an analysis unchanged."""
    print("\n--- PERF-4 F2: the cached parse is pristine and stays pristine ---")
    code = "float g = $k * 2.0;\n@OUT = (g > 1.0) ? @A : @B;"
    tex_lazy.clear_lazy_memo()
    pristine = tex_lazy._pristine_program(code)
    before = repr(pristine)
    for v in (0.0, 1.0, 4.0):
        tex_lazy.lazy_required_bindings(code, {"k": v})
    r.ok("three evaluations left the cached parse byte-identical") \
        if repr(tex_lazy._pristine_program(code)) == before else \
        r.fail("PERF-4 lazy sharing",
               "the cached parse was mutated, so every later value of this source would "
               "inherit the previous tick's literals")

    for i in range(tex_lazy._PARSE_MEMO_MAX + 20):
        tex_lazy._pristine_program(f"@OUT = vec4({i}.0);")
    over = len(tex_lazy._parse_memo)
    tex_lazy.clear_lazy_memo()
    r.ok(f"the parse memo is bounded at {tex_lazy._PARSE_MEMO_MAX} (held {over}) and clears") \
        if over <= tex_lazy._PARSE_MEMO_MAX and not tex_lazy._parse_memo else \
        r.fail("PERF-4 lazy memo bound",
               f"{over} entries with a cap of {tex_lazy._PARSE_MEMO_MAX}, "
               f"{len(tex_lazy._parse_memo)} left after clear_lazy_memo()")

    # A source that does not parse is not cached, and still answers None (keep everything).
    broken = "@OUT = ;;;"
    r.ok("an unparseable source answers None and caches no entry") \
        if tex_lazy.lazy_required_bindings(broken, {}) is None \
        and (broken, tex_lazy._profile_key()) not in tex_lazy._parse_memo else \
        r.fail("PERF-4 lazy parse error",
               "a source that does not parse either answered something or was cached")


@isolated_analysis
def test_perf4_lazy_oracle_sensitive_rows(r: SubTestResult):
    """NOT VACUOUS. Each `_LAZY_SENSITIVE` source really does answer differently per value,
    except the one chosen to prove the never-sever rule, which must answer the SAME."""
    print("\n--- PERF-4 F2: the corpus contains value-sensitive programs ---")
    for label, code in sorted(_LAZY_SENSITIVE.items()):
        answers = {tex_lazy.lazy_required_bindings(code, p) for p in _valuations_for(code)}
        if label == "never_severed":
            kept = all(a is not None and "A" in a for a in answers)
            r.ok("never_severed: @A survives `* 0.0` under every valuation (invariant 11)") \
                if kept else \
                r.fail("PERF-4 lazy sensitivity",
                       "`@A * 0.0` severed @A — the lazy analysis may only OVER-approximate")
        else:
            r.ok(f"{label}: {len(answers)} distinct answers across the valuations") \
                if len(answers) > 1 else \
                r.fail("PERF-4 lazy sensitivity",
                       f"{label} answers the same for every value — it is not exercising the "
                       f"value dependence it was chosen for")


# ── F4: the halo scan ────────────────────────────────────────────────────────

@isolated_analysis
def test_perf4_halo_answers_are_identical(r: SubTestResult):
    """Every folded program answers what the pre-change two-pass scan answered."""
    print("\n--- PERF-4 F4: the single-traversal halo scan vs the pre-change one ---")
    rows = _corpus()
    bad, checked = [], 0
    for label, code in rows:
        for params in _valuations_for(code):
            a, b = _folded(code, params), _folded(code, params)
            if a is None or b is None:
                continue
            got, want = tex_roi._has_ungrounded_halo(a), _base_has_ungrounded_halo(b)
            checked += 1
            if got != want:
                bad.append(f"{label} params={params}: {want} -> {got}")
    if bad:
        r.fail("PERF-4 halo identity",
               f"{len(bad)} of {checked} verdicts moved; first 3: {bad[:3]}")
    else:
        r.ok(f"{checked} halo verdicts over {len(rows)} sources are unchanged")


@isolated_analysis
def test_perf4_halo_corpus_is_not_vacuous(r: SubTestResult):
    """NOT VACUOUS. The halo corpus contains both verdicts, and each shape answers what its
    name claims — otherwise "identical over the corpus" would be a statement about nothing."""
    print("\n--- PERF-4 F4: the halo corpus exercises both verdicts ---")
    expect = {"grounded": False, "case1_vardecl": True, "case1_loop": True,
              "case2_direct": True, "case2_nested": True, "case2_in_if": True,
              "named_no_halo": False, "written_unread": False}
    bad = []
    for label, code in sorted(_HALO_SHAPES.items()):
        prog = _folded(code, {"k": 0.75})
        if prog is None:
            bad.append(f"{label}: did not fold")
            continue
        got = tex_roi._has_ungrounded_halo(prog)
        if got != expect[label]:
            bad.append(f"{label}: expected {expect[label]}, got {got}")
    if bad:
        r.fail("PERF-4 halo corpus", "; ".join(bad))
    else:
        r.ok(f"{len(expect)} halo shapes answer as named "
             f"({sum(expect.values())} blocked, {len(expect) - sum(expect.values())} not)")


@isolated_analysis
def test_perf4_the_halo_scan_visits_each_node_once(r: SubTestResult):
    """RED-FIRST, the structural gate. One descent per AST node per scan, not two.

    Spies on `tex_roi.iter_child_nodes` — the module attribute the scan resolves at call time
    — while `_has_ungrounded_halo` runs, and counts the descents PER NODE. The program is
    chosen to carry no halo at all, so the name-intersection pass (`_collect_read_names`,
    which is conditional and descends a different subset) never runs and the count is the
    halo scan's alone. The expected total is DERIVED, not pinned: it is the program's node
    count, computed here by an independent walk.

    On the base sha this reads 2 descents per node for the two traversals plus one more per
    ancestor for `_subtree_has_halo`'s re-descent."""
    print("\n--- PERF-4 F4: one descent per node per scan ---")
    code = ("vec4 base = @A * 2.0;\n"
            "float k = 0.5;\n"
            "@OUT = vec4(mix(base.rgb, @B.rgb, k) + vec3(0.1, 0.2, 0.3), 1.0);\n")
    program = _folded(code, {})
    if program is None:
        r.fail("PERF-4 halo scan-count", "the probe program did not fold")
        return

    def _node_count(node):
        return 1 + sum(_node_count(ch) for ch in iter_child_nodes(node))

    expected = sum(_node_count(s) for s in program.statements)

    real = tex_roi.iter_child_nodes
    seen: dict = {}

    def counting(node):
        seen[id(node)] = seen.get(id(node), 0) + 1
        return real(node)

    tex_roi.iter_child_nodes = counting
    try:
        verdict = tex_roi._has_ungrounded_halo(program)
    finally:
        tex_roi.iter_child_nodes = real

    total, worst = sum(seen.values()), max(seen.values(), default=0)
    if verdict is not False:
        r.fail("PERF-4 halo scan-count",
               "the probe program was expected to be unblocked (no halo op anywhere); the "
               "count below would then include the conditional read pass")
        return
    if worst != 1 or total != expected:
        r.fail("PERF-4 halo scan-count",
               f"{total} descents over {len(seen)} nodes (worst node descended {worst}x); "
               f"one scan of this program is exactly {expected} descents, one per node")
    else:
        r.ok(f"{expected} nodes, {total} descents, no node descended twice")


@isolated_analysis
def test_perf4_halo_mutants_are_caught(r: SubTestResult):
    """MUTATION, both rules. Break the merged scan's bottom-up halo propagation, then case
    (1)'s grounded/ungrounded distinction, and require the corpus to catch each — then require
    the SHIPPED scan to agree with the oracle on the very same rows."""
    print("\n--- PERF-4 F4 mutation: each rule of the merged scan is load-bearing ---")
    rows = [(f"halo:{k}", v) for k, v in _HALO_SHAPES.items()]
    for flag in ("shallow_value", "ignore_ungrounded"):
        caught = []
        for label, code in rows:
            for params in ({}, {"k": 0.75}):
                a, b = _folded(code, params), _folded(code, params)
                if a is None or b is None:
                    continue
                if _mutant_has_ungrounded_halo(a, **{flag: True}) != \
                        _base_has_ungrounded_halo(b):
                    caught.append(label)
        r.ok(f"{flag}: caught on {len(set(caught))} corpus shape(s) — the rule is "
             f"load-bearing") if caught else \
            r.fail("PERF-4 halo mutation",
                   f"{flag} changed no verdict anywhere in the corpus — the corpus does not "
                   f"exercise the rule, so 'identical' says nothing about it")

    bad = []
    for label, code in rows:
        for params in ({}, {"k": 0.75}):
            a, b = _folded(code, params), _folded(code, params)
            if a is None or b is None:
                continue
            if tex_roi._has_ungrounded_halo(a) != _base_has_ungrounded_halo(b):
                bad.append(label)
    r.ok("the shipped scan agrees with the oracle on every one of those rows") if not bad else \
        r.fail("PERF-4 halo mutation", f"the shipped scan disagrees on: {sorted(set(bad))}")
