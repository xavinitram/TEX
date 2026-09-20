"""PERF-1 — the ROI walk parses a source once, and answers exactly what it answered before.

WHAT CHANGED. `tex_roi._walk` memoizes on `(sha256(code), param VALUES, string wires)`, and
that key is right: the walk's answer genuinely depends on the values (a `$sigma` in a halo
position changes the halo; `mix(@A, @B, $k)` with `k = 0` folds `@B` away, which is what
`fold_erased` exists to catch). But a slider therefore missed the memo on every tick and paid
a full `Lexer.tokenize` + `Parser.parse` for a program whose SOURCE had not moved. The parse
is now memoized per source (`tex_roi._pristine_program`) and each fold works on its own
`ast_nodes.clone_tree` copy, so the fold still runs per value — on a reused parse.

WHY THIS FILE IS SHAPED LIKE A DERIVATION ORACLE. The change is behaviour-preserving or it is
nothing, and "behaviour" here is five values (`reads`, `blocked`, `halo`, `fold_erased`,
`region_dep`) over an open set of sources and an open set of parameter valuations. So the
PRE-CHANGE implementation is kept below, verbatim in the one line that matters — a fresh
`parse_and_split` per call — and the two are run against each other over every shipped
`examples/*.tex`, the ten-stage host-demo comp, and a hand-written corpus of the shapes where
a VALUE is known to move the answer. `_ORACLE_SENSITIVE_ROWS` pins that the corpus really does
contain such shapes, so the comparison cannot pass by being vacuous.

BOTH DIRECTIONS. `test_perf1_the_clone_is_load_bearing` breaks the clone (hands the fold the
memoized parse itself) and requires the oracle to NOTICE — a test that cannot fail is not a
test. `test_perf1_a_source_is_parsed_once` is the red-first half: it fails on the base sha,
where N values cost N lexes.

PORTABILITY: CPU, no ComfyUI, no CUDA, no compiler, no numpy, no timing assertion.
"""
import glob
import math
import os

from helpers import *

from TEX_Wrangle import tex_roi
from TEX_Wrangle.tex_cache import parse_and_split
from TEX_Wrangle.tex_compiler import ast_nodes
from TEX_Wrangle.tex_compiler.ast_nodes import Assignment, NumberLiteral
from TEX_Wrangle.tex_compiler.optimizer import _fold_all, _propagate_literal_locals
from TEX_Wrangle.tex_lazy import _fp32, _substitute_params
from TEX_Wrangle.tex_marshalling import sigil_names

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _drop_token_offer(code: str) -> None:
    """PERF-5: consume the token stream `sigil_names` offers to the next parse of `code`, so a
    priming call in this file donates no lex to a row that is counting lexes."""
    from TEX_Wrangle.tex_compiler.lexer import claim_tokens
    claim_tokens(code, dotted_bindings=True)


# ── the pre-change implementation, kept as the oracle ────────────────────────

def _base_fold_program(code: str, param_values: dict):
    """`tex_roi._fold_program` as it stood at the base sha: a FRESH parse every call.

    Everything after the parse is the shipped code — this file imports the same
    `_substitute_params` / `_fold_all` / `_propagate_literal_locals` the module does, so the
    oracle differs from the implementation in exactly the thing under test and drifts with
    the fold if the fold ever changes."""
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
        program.statements = stmts
    return program


def _base_walk(code: str, param_values: dict, binding_types=None):
    """`tex_roi._walk`'s body at the base sha, UNMEMOIZED, over `_base_fold_program`."""
    try:
        program = _base_fold_program(code, param_values)
        reads: dict = {}
        state = {"blocked": False, "halo": 0}
        for stmt in program.statements:
            tex_roi._accumulate(stmt, 0, reads, state)
        blocked = state["blocked"] or tex_roi._has_ungrounded_halo(program)
        written = {n for n in (tex_roi._write_target_name(s.target, bindings_only=True)
                               for s in program.statements if isinstance(s, Assignment))
                   if n is not None}
        return (reads, blocked, state["halo"],
                tex_roi._referenced_at_bindings(code) - written - set(reads),
                tex_roi.region_dependent(program, binding_types, code))
    except Exception:
        return None


def _canon(walked):
    """The walk's five-tuple as a comparable, hashable value (`_Reads` has no `__eq__`)."""
    if walked is None:
        return None
    reads, blocked, halo, erased, region_dep = walked
    return (tuple(sorted(((n, e.narrow_reach, e.has_narrow, e.whole, e.has_whole)
                          for n, e in reads.items()), key=lambda t: t[0])),
            bool(blocked), int(halo), tuple(sorted(erased)), bool(region_dep))


# ── the corpus ───────────────────────────────────────────────────────────────

#: Sources whose walk is KNOWN to move with a value. Named, not discovered, so the oracle's
#: sensitivity is a pin rather than a hope: `_ORACLE_SENSITIVE_ROWS` requires each of these to
#: produce at least two distinct answers across `_VALUATIONS`.
_SENSITIVE = {
    # ROI-2's own erasure case: `k = 0` folds `@B` out of `reads` and into `fold_erased`.
    "mix_erasure": "@OUT = mix(@A, @B, $k);",
    # A halo RADIUS carried by a parameter: the cook halo is 3*sigma, so the value IS the reach.
    "halo_radius": "@OUT = gauss_blur(@IN, $sigma);",
    # A parameter-gated ternary over two halo ops: which reach survives depends on the value.
    "gated_halo": "@OUT = ($k > 0.5) ? gauss_blur(@A, 2.0) : gauss_blur(@A, 8.0);",
    # A parameter reaching a radius through a local, so the fold must propagate to resolve it.
    "halo_via_local": "float r = $sigma * 2.0;\n@OUT = gauss_blur(@IN, r);",
}

#: The valuations every corpus source is walked under. Each one is a shape that has moved an
#: answer here before: the fold-erasing 0 and 1, a NaN (which must not become a radius), an
#: int-valued float against a true int (they type-tag differently in `_param_key`), a bool,
#: a negative, and two radii that differ in their ceil.
_VALUATIONS = (0.0, 1.0, 0.5, 2.0, 2, True, -1.0, 3.0, 7.0, 0.25, float("nan"),
               float("inf"), 1e-8)


def _corpus():
    """`(label, code)` for every shipped example, the host-demo comp and `_SENSITIVE`."""
    rows = [(f"sensitive:{k}", v) for k, v in _SENSITIVE.items()]
    demo = os.path.join(_ROOT, "examples", "host_demo.py")
    if os.path.exists(demo):
        import importlib.util
        spec = importlib.util.spec_from_file_location("_perf1_host_demo", demo)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        rows += [(f"comp:{n}", c) for n, c, _p in mod._COMP_STAGES]
    for path in sorted(glob.glob(os.path.join(_ROOT, "examples", "*.tex"))):
        with open(path, "r", encoding="utf-8") as fh:
            rows.append(("example:" + os.path.basename(path), fh.read()))
    return rows


def _valuations_for(code: str):
    """Every `$name` in the source bound to each value of `_VALUATIONS`, plus the empty dict.

    The empty dict is not a valuation like the others: `_fold_program` runs the optimizer's
    fold only when at least one foldable param is supplied, so "no params" is its own program
    shape and the oracle has to cover it."""
    names = sorted(sigil_names(code)[1])
    out = [{}]
    for v in _VALUATIONS:
        out.append({n: v for n in names})
    if len(names) > 1:                       # one param moved while the others hold still
        out.append(dict({n: 0.5 for n in names}, **{names[0]: 0.0}))
        out.append(dict({n: 0.5 for n in names}, **{names[-1]: 8.0}))
    return out


_STRING_WIRES = ({}, {"S": TEXType.STRING}, {"A": TEXType.STRING, "S": TEXType.STRING})


# ── the rows ─────────────────────────────────────────────────────────────────

def test_perf1_walk_answers_are_identical(r: SubTestResult):
    """Every `(source, valuation, string-wire map)` answers what the pre-change walk answers."""
    print("\n--- PERF-1: the memoized-parse walk vs the pre-change walk ---")
    rows = _corpus()
    bad, checked = [], 0
    for label, code in rows:
        for params in _valuations_for(code):
            for types in _STRING_WIRES:
                got = _canon(tex_roi._walk(code, params, types))
                want = _canon(_base_walk(code, params, types))
                checked += 1
                if got != want:
                    bad.append(f"{label} params={params} types={sorted(types)}: "
                               f"{want!r} -> {got!r}")
    if bad:
        r.fail("PERF-1 walk identity",
               f"{len(bad)} of {checked} answers moved; first 3: {bad[:3]}")
    else:
        r.ok(f"{checked} walk answers over {len(rows)} sources are unchanged")


def test_perf1_a_source_is_parsed_once(r: SubTestResult):
    """RED-FIRST. Many parameter values over one source cost ONE lex and ONE parse.

    This is the whole ask as an assertion, and it fails on the base sha (where each value
    re-lexes and re-parses). It counts through the real front end rather than through
    `_pristine_program`, so a fix that moved the parse somewhere else would not satisfy it."""
    print("\n--- PERF-1: one lex + one parse per distinct source ---")
    from TEX_Wrangle.tex_compiler import lexer as _lx, parser as _ps
    code = "@OUT = vec4(@IN.rgb * $exposure, 1.0);\n@B = gauss_blur(@A, $sigma);"
    tex_roi.clear_roi_memo()
    # `_referenced_at_bindings` lexes through `sigil_names`, which memoizes per source: prime
    # it, so what is counted below is the ROI fold's own front-end work and nothing else.
    # PERF-5: that scan now also OFFERS its token stream to the next `parse_and_split` of the
    # same source, so priming it would donate this row's one lex and the count would read 0.
    # Drop the offer — the ROI fold's own lex is what this row is about, and leaving it would
    # make PERF-1's gate depend on PERF-5's mechanism.
    sigil_names(code)
    _drop_token_offer(code)
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
            tex_roi._walk(code, {"exposure": 1.0 + i * 0.01, "sigma": 1.0 + i * 0.1})
    finally:
        _lx.Lexer.tokenize, _ps.Parser.parse = lex0, parse0
    if (n["lex"], n["parse"]) != (1, 1):
        r.fail("PERF-1 parse-once", f"24 parameter values cost {n['lex']} lex(es) and "
                                    f"{n['parse']} parse(s); expected 1 and 1")
    else:
        r.ok("24 parameter values over one source cost 1 lex and 1 parse")

    # ... and a SECOND source is still parsed, so the memo is a cache and not a swallow.
    n["lex"] = n["parse"] = 0
    other = "@OUT = vec4(@IN.rgb + vec3($lift), 1.0);"
    sigil_names(other)
    _drop_token_offer(other)
    _lx.Lexer.tokenize, _ps.Parser.parse = lex, parse
    try:
        tex_roi._walk(other, {"lift": 0.1})
    finally:
        _lx.Lexer.tokenize, _ps.Parser.parse = lex0, parse0
    r.ok("a second source is parsed once more (the memo keys on the source)") \
        if (n["lex"], n["parse"]) == (1, 1) else \
        r.fail("PERF-1 parse-once", f"a NEW source cost {n['lex']}/{n['parse']} lex/parse, "
                                    f"expected 1/1 — the memo is answering for the wrong source")


def test_perf1_the_clone_is_load_bearing(r: SubTestResult):
    """MUTATION. Hand the fold the memoized parse ITSELF and the oracle must notice.

    The memo holds a PRISTINE parse and the fold rewrites in place, so without the copy the
    second value folds a tree the first value already folded — literals from the previous
    scrub tick, permanently. This row proves the corpus and the comparison would catch that,
    which is the only reason to believe the passing row above."""
    print("\n--- PERF-1 mutation: the fold's copy is load-bearing ---")
    code = _SENSITIVE["mix_erasure"]
    real = tex_roi.clone_tree
    tex_roi.clear_roi_memo()
    tex_roi.clone_tree = lambda p: p          # the broken version: no copy at all
    try:
        got = [_canon(tex_roi._walk(code, {"k": v})) for v in (0.0, 1.0, 0.5)]
    finally:
        tex_roi.clone_tree = real
        tex_roi.clear_roi_memo()
    want = [_canon(_base_walk(code, {"k": v})) for v in (0.0, 1.0, 0.5)]
    r.ok("a fold without the copy is CAUGHT by the oracle (the copy is load-bearing)") \
        if got != want else \
        r.fail("PERF-1 mutation", "removing the AST copy changed no answer — the oracle is "
                                  "not sensitive to the thing this change rests on")

    # And the other direction: with the copy restored, those same three answers are right.
    tex_roi.clear_roi_memo()
    fixed = [_canon(tex_roi._walk(code, {"k": v})) for v in (0.0, 1.0, 0.5)]
    r.ok("with the copy restored the same three valuations agree with the oracle") \
        if fixed == want else \
        r.fail("PERF-1 mutation", f"restored walk still disagrees: {want!r} -> {fixed!r}")


def test_perf1_oracle_sensitive_rows(r: SubTestResult):
    """NOT VACUOUS. Each `_SENSITIVE` source really does answer differently per value.

    If every corpus answer were value-INDEPENDENT, the identity row above would pass for a
    walk that ignored its parameters entirely. This is the row that forbids that reading."""
    print("\n--- PERF-1: the corpus contains value-sensitive programs ---")
    for label, code in sorted(_SENSITIVE.items()):
        answers = {_canon(tex_roi._walk(code, p)) for p in _valuations_for(code)}
        r.ok(f"{label}: {len(answers)} distinct walk answers across the valuations") \
            if len(answers) > 1 else \
            r.fail("PERF-1 sensitivity", f"{label} answers the same for every value — it is "
                                         f"not exercising the value dependence it was chosen for")


def test_perf1_the_memo_hands_out_no_shared_ast(r: SubTestResult):
    """The memoized parse is never handed to a caller, and survives a fold unchanged.

    `_fold_program` is called by `_walk`, `frame_window` and `batch_sliceable`; each mutates
    what it is given. This row pins the entry's identity and its content: a fold must not be
    able to reach the cached tree."""
    print("\n--- PERF-1: the cached parse is pristine and stays pristine ---")
    code = "float r = $sigma * 2.0;\n@OUT = gauss_blur(@IN, r) + vec4($k);"
    tex_roi.clear_roi_memo()
    pristine = tex_roi._pristine_program(code)
    before = repr(pristine)
    for v in (0.0, 1.0, 4.0):
        folded = tex_roi._fold_program(code, {"sigma": v, "k": v})
        if folded is pristine or folded.statements is pristine.statements:
            r.fail("PERF-1 sharing", "the fold was handed the memo entry itself")
            return
        tex_roi.frame_window(code, {"sigma": v, "k": v})
        tex_roi.batch_sliceable(code, {"sigma": v, "k": v})
    r.ok("three folds and the two other consumers left the cached parse byte-identical") \
        if repr(tex_roi._pristine_program(code)) == before else \
        r.fail("PERF-1 sharing", "the cached parse was mutated by a fold — every later value "
                                 "of this source would inherit the previous tick's literals")

    # The bound holds, and `clear_roi_memo` empties it with the rest.
    for i in range(tex_roi._PARSE_MEMO_MAX + 20):
        tex_roi._pristine_program(f"@OUT = vec4({i}.0);")
    over = len(tex_roi._parse_memo)
    tex_roi.clear_roi_memo()
    r.ok(f"the parse memo is bounded at {tex_roi._PARSE_MEMO_MAX} (held {over}) and clears") \
        if over <= tex_roi._PARSE_MEMO_MAX and not tex_roi._parse_memo else \
        r.fail("PERF-1 memo bound", f"{over} entries with a cap of {tex_roi._PARSE_MEMO_MAX}, "
                                    f"{len(tex_roi._parse_memo)} left after clear_roi_memo()")


def test_perf1_clone_tree_is_a_faithful_copy(r: SubTestResult):
    """`ast_nodes.clone_tree` reproduces a parse: same shape, no shared mutable state.

    The three rules its docstring states, each as a check — because each is a thing a
    plausible cheaper copy gets wrong, and none of them would show up as a parse error."""
    print("\n--- PERF-1: clone_tree is indistinguishable from a second parse ---")
    code = ("float lut[3] = {1.0, 2.0, 3.0};\n"
            "f$gain [min: 0, max: 2, label: \"Gain\"];\n"
            "float helper(float a, float b) { return a * b; }\n"
            "float x = 0.0;\n"
            "for (int i = 0; i < 3; i = i + 1) { x = x + lut[i]; }\n"
            "if (v > 0.5) { x = helper(x, $gain); } else { x = x * 2.0; }\n"
            "@OUT = vec4(gauss_blur(@IN, 2.0).rgb * x, 1.0);\n")
    a = parse_and_split(code, {})
    b = ast_nodes.clone_tree(a)
    r.ok("the copy reprs identically to the tree it was made from") if repr(a) == repr(b) else \
        r.fail("PERF-1 clone shape", "clone_tree produced a different tree")

    shared = []
    seen = set()

    def walk(x, y, path):
        if x is y:
            shared.append(path)
            return
        if type(x) is not type(y):
            shared.append(path + " (type)")
            return
        if id(x) in seen:
            return
        seen.add(id(x))
        for name in (f.name for f in __import__("dataclasses").fields(type(x))):
            vx, vy = getattr(x, name), getattr(y, name)
            if isinstance(vx, ast_nodes.ASTNode):
                walk(vx, vy, f"{path}.{name}")
            elif isinstance(vx, list):
                if vx is vy and vx:
                    shared.append(f"{path}.{name} (list)")
                for i, (ex, ey) in enumerate(zip(vx, vy)):
                    if isinstance(ex, ast_nodes.ASTNode):
                        walk(ex, ey, f"{path}.{name}[{i}]")
            elif isinstance(vx, ast_nodes.SourceLoc) and vx is vy:
                shared.append(f"{path}.{name} (loc)")
            elif isinstance(vx, dict) and vx is vy and vx:
                shared.append(f"{path}.{name} (dict)")

    walk(a, b, "program")
    r.ok("no node, list, dict or SourceLoc is shared between the tree and its copy") \
        if not shared else \
        r.fail("PERF-1 clone sharing", f"{len(shared)} shared slot(s): {shared[:4]}")

    # Mutating the copy — which is exactly what the fold does — leaves the original alone.
    before = repr(a)
    b.statements = _fold_all(b.statements)
    for stmt in b.statements:
        _substitute_params(stmt, {"gain": NumberLiteral(value=9.0, is_int=False)})
    r.ok("folding and substituting into the copy left the original untouched") \
        if repr(a) == before else \
        r.fail("PERF-1 clone isolation", "a rewrite of the copy reached the original tree")

    # Aliasing INSIDE the tree survives the copy (the id-keyed memo's reason for existing).
    lit = NumberLiteral(value=1.0, is_int=False)
    twice = ast_nodes.Program(statements=[
        ast_nodes.ExprStatement(expr=lit), ast_nodes.ExprStatement(expr=lit)])
    copy = ast_nodes.clone_tree(twice)
    r.ok("a node reachable twice is still one node in the copy") \
        if copy.statements[0].expr is copy.statements[1].expr else \
        r.fail("PERF-1 clone aliasing", "clone_tree unshared an aliased node, so an in-place "
                                        "rewrite would land on one occurrence and not the other")


def test_perf1_nan_and_inf_do_not_become_a_radius(r: SubTestResult):
    """A non-finite parameter in a halo position stays unresolved, before and after.

    Called out on its own because it is the valuation most likely to differ between a fresh
    parse and a reused one if the fold ever grew a cache of its own: `ceil(nan)` raises, and
    a walk that answered `blocked=False` with an unusable reach would narrow a cook it cannot
    narrow."""
    print("\n--- PERF-1: NaN / inf in a halo radius ---")
    code = _SENSITIVE["halo_radius"]
    for v in (float("nan"), float("inf"), float("-inf")):
        got, want = _canon(tex_roi._walk(code, {"sigma": v})), \
            _canon(_base_walk(code, {"sigma": v}))
        label = "nan" if math.isnan(v) else repr(v)
        r.ok(f"sigma={label}: {'blocked' if got and got[1] else 'unblocked'}, unchanged") \
            if got == want else \
            r.fail("PERF-1 non-finite", f"sigma={label}: {want!r} -> {got!r}")
