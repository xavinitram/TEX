"""GATE-47 (TRK-208, shape (a)) — no NEW wall-clock comparison enters an assertion without
either `@pytest.mark.timing`/`@pytest.mark.slow` or a reviewed allowlist entry.

`CI-461`'s own audit named the structural gap this closes: a wall-clock microbenchmark
compared a cached lookup against a bare `import torch` and was green on every local gate
tier (cheap/touched/full — none of them trace coverage) and red on every CI Python version,
because GitHub Actions runs the whole suite with `--cov=TEX_Wrangle` always on and
`-m 'not slow'` does NOT deselect `timing`. Coverage's line tracer taxes an extra
Python-level function call far more than it taxes one `IMPORT_NAME` bytecode, so a "not
slower than X" claim can invert under tracing even though the code changed nothing. That
fix replaced the one test; nothing stopped a NEW instance of the same class from shipping.
This is the ratchet an embedding host asked for.

WHAT COUNTS AS THE DEFECT SHAPE
--------------------------------
A wall-clock READING alone is not the defect — `test_matrix_benchmarks` (`test_integration.py`)
times several kernels and reports them via `r.ok(f"...{elapsed:.1f}ms")` with no comparison at
all, and nothing here has any reason to touch it: an *informational* timing that never gates
pass/fail cannot invert into a false red under tracing. The defect is a COMPARISON against a
bound, inside something that decides pass/fail — an `assert`, or the `test` of an `if` (through
any chain of `and`/`or`/`not`) whose outcome reaches `r.ok`/`r.fail` — where an operand reads as
duration-shaped (`elapsed`, `dt`, `_ms`, `ratio`, `deadline`, ...; see `_DURATION_RE`) AND the
enclosing test function's OWN body captures a wall-clock reading itself (`time.perf_counter()`,
`time.time()`, `timeit.*`, `.monotonic()`, `.process_time()`; see `_WALLCLOCK_SRC_RE`). Both
conditions matter: the second is what ties a flagged comparison to the actual coverage-tracing
mechanism — a duration MEASURED IN THIS PROCESS is taxed by the tracer; a duration handed back
from a **child** process (a `subprocess.run(...)` probe that prints its own `time.perf_counter()`
reading as JSON, e.g. this file's own `test_v0422_redos.py`-style ReDoS guards) is not, because
`pytest-cov` traces this process, never an arbitrary child it merely launched. That is a
deliberate, honest boundary of this scanner (the G4 lesson `tools/gate.py` already names for
its own import-matching walk): a subprocess-measured bound is a structurally DIFFERENT, already
coverage-immune shape, so it is correctly left alone rather than forced through the same
allowlist for a risk it does not carry. A `while`/`for` loop condition (a poll-until-deadline
loop, e.g. `test_v034_io1.py`'s `while ... time.perf_counter() < deadline`) is excluded the same
way — the comparison there governs how long a wait spins, not whether the test passes, so it is
outside `_is_guard_compare`'s reach by construction (only an `If`/`Assert` test counts, walked up
through boolean connectives only).

THE ALLOWLIST
-------------
`_ALLOWLIST` names every (file, test) this scanner flags at the base sha it landed on, each
with why it is not yet marked. It is a REVIEWED list, not a way to make a red go away quietly:
a stale entry (the scanner no longer flags that test — the code moved on) is itself a failure
below, exactly like `tools/gate.py`'s own known-red allowlist. Every entry is a candidate for
`@pytest.mark.timing` in a future pass, not a permanent exemption.
"""
import ast
import glob
import os
import re
import tempfile

from helpers import SubTestResult

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

#: Duration-shaped identifier fragments. Deliberately SUBSTRING matches (no `\b...\b` on most
#: of them): real variable names are snake_case compounds (`cached_elapsed`, `fresh_elapsed`,
#: `drain_tail`), so a whole-word anchor misses exactly the shape this exists to catch — proven
#: by `test_gate47_ratchet_catches_the_v046_ci_failure_shape` below, which reds under a
#: whole-word version of this pattern and passes under this one. `dt` alone stays whole-word
#: (`\bdt\b`): as a bare two-letter name it is common ONLY as "delta time"; as a substring it
#: would hit too much (e.g. "width", "edit").  Bare `ms` is deliberately NOT a pattern — it is
#: an extremely common variable name for a NON-wall-clock duration this project already uses
#: (e.g. `test_prof462_device_honest.py`'s fake-device-event millisecond readings), and would
#: flood this ratchet with the exact class of false positive `_WALLCLOCK_SRC_RE`'s co-condition
#: exists to filter out anyway.
_DURATION_RE = re.compile(
    r"(?i)elapsed|\bdt\b|duration|latency|runtime|drained?|speedup|ratio|deadline|"
    r"wall_s|_ms\b|_sec\b|_secs\b|p50|p95|p99"
)

#: A wall-clock reading taken IN THIS PROCESS. Never `.sleep(` (a delay, not a measurement).
#: FIX-GATE G1 (B4#3) widened this past the two `timeit.*` spellings the module docstring
#: already claimed but the pattern did not actually cover: `timeit.repeat(...)` (a second,
#: equally-common entry point) and `Timer(...).timeit(...)` (a bound method call, where the
#: literal substring "timeit." never appears before ".timeit(" because the `Timer` instance
#: sits in a local variable) -- the bare `.timeit(` alone closes that one.
_WALLCLOCK_SRC_RE = re.compile(
    r"perf_counter\(\)|time\.time\(\)|timeit\.default_timer\(\)|timeit\.timeit\(|"
    r"timeit\.repeat\(|timeit\.Timer\(|\.timeit\(|"
    r"\.monotonic\(\)|\.process_time\(\)"
)

_MARKER_RE = re.compile(r"mark\.timing|mark\.slow")

#: HOUSE-50/H1 — a call to the engine's real per-pixel executor. Deliberately narrow (any
#: `.cook(` attribute call): what distinguishes "a program actually ran and got timed" from
#: "the profiler's table was SEEDED with a literal constant" (`P.record(key, 10.0, ...)`,
#: this file's own `test_v031_prof1_predicts_an_unseen_resolution`; `test_v032_checkpoint.py`'s
#: CACHE-7 fallback rows via `_profile.record_stages(...)`) is whether a real cook ran --
#: seeding-and-reading-back is pure, deterministic arithmetic on a number the test itself
#: chose, never a measurement of anything, and must stay green.
_COOK_CALL_RE = re.compile(r"\.cook\(")

#: A read of PROF-1's own recorded-cost table (`tex_runtime/profile.py`'s `snapshot`/
#: `stage_snapshot`/`stage_costs`), populated by `profile.measure`'s own
#: `time.perf_counter()` pair (`profile.py`: `self._t0 = time.perf_counter()` /
#: `record(self.key, (time.perf_counter() - self._t0) * 1000.0, ...)`). That is an
#: in-process wall-clock reading exactly like a bare `perf_counter()` call -- except the
#: call site lives in a DIFFERENT module than the test file this scanner walks, so
#: `_WALLCLOCK_SRC_RE` (a same-file textual scan) cannot see it. Paired with `_COOK_CALL_RE`
#: above (see `scan_source`) rather than trusted alone, for the seeding reason given there.
_PROFILER_READ_RE = re.compile(r"\.snapshot\(\)|\.stage_snapshot\(|\.stage_costs\(")

#: Reviewed at GATE-47 (v0.47.0), against the tree at `b7a4e3d`. Keyed by
#: `(tests-relative posix path, function name)`. An entry here is a decision made out loud —
#: not the same as `@pytest.mark.timing`, which is the ask this ratchet is nudging toward.
_ALLOWLIST: dict = {
    ("tests/test_codegen_optimizer.py", "test_optimizer_pure_fn_cse_licm"):
        "a ~1500-node optimize() pass must finish inside a generous 5.0s absolute budget "
        "(a hang/superlinearity guard, not a tight perf claim) -- pre-existing, named by "
        "CI-461's own wall-clock audit, not yet marked",
    ("tests/test_pace45_pacing.py", "test_pace45_cuda_pacing_bit_exact_and_repro"):
        "CUDA-gated (self-skips on a CPU-only box/CI runner): compares a cancel's observed "
        "latency against the same box's own measured uncancelled runtime -- pre-existing, "
        "named by CI-461's own wall-clock audit, not yet marked",
    ("tests/test_v018_precision.py", "test_c1_amplification_gate"):
        "a CPU-fine superlinearity guard on the F2 static walk (2000ms absolute budget over "
        "a 2000-deep / 2^26-fan-out program pair) -- pre-existing, named by CI-461's own "
        "wall-clock audit, not yet marked",
}


def _parent_map(tree: ast.AST) -> dict:
    parents: dict = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _is_guard_node(node: ast.AST, parents: dict) -> bool:
    """True when `node` (a `Compare` OR a bare `Call`, FIX-GATE G1/B4#4) is, or sits under,
    through any chain of `and`/`or`/`not`, the `test` of an `if` or an `assert` -- the two
    shapes that decide pass/fail. A `while`/`for` test walks up to a node type this never
    returns True for, so a poll-until-deadline loop is excluded by construction, not by a
    special case. (Named `_is_guard_compare` before G1 widened it past `Compare` alone;
    kept as an alias below for anything that still spells the old name.)"""
    cur = node
    while True:
        p = parents.get(cur)
        if p is None:
            return False
        if isinstance(p, ast.BoolOp) and cur in p.values:
            cur = p
            continue
        if isinstance(p, ast.UnaryOp) and cur is p.operand:
            cur = p
            continue
        if isinstance(p, ast.If) and cur is p.test:
            return True
        if isinstance(p, ast.Assert) and cur is p.test:
            return True
        return False


_is_guard_compare = _is_guard_node   # pre-G1 name, same predicate widened to cover Call too


def _is_ternary_guard_node(node: ast.AST, parents: dict) -> bool:
    """HOUSE-50/H1 -- like `_is_guard_node`, but ALSO recognizes a ternary (`ast.IfExp`)
    `test` as a guard position: `r.ok(...) if COND else r.fail(...)`, the `SubTestResult`
    idiom this project's own test suite spells almost every assertion with, instead of an
    `if`/`else` STATEMENT (`test_v031_prof1_per_stage_breakdown`'s own
    `r.ok(...) if heavy == "1" and hms > 2.0 * sms else r.fail(...)` is exactly this shape).

    Kept SEPARATE from `_is_guard_node` rather than folded into it: widening the GENERAL
    scanner to walk into every ternary in the whole suite is a much bigger, unaudited change
    (thousands of pre-existing, already-green ternary comparisons this ratchet has never
    looked at) than teaching it the one concrete shape this ask names. Used only by the
    profiler-stage-relative-comparison check in `scan_source`, which already narrows to the
    handful of functions that both ran a real cook (`_COOK_CALL_RE`) and read the cost table
    back (`_PROFILER_READ_RE`)."""
    cur = node
    while True:
        p = parents.get(cur)
        if p is None:
            return False
        if isinstance(p, ast.BoolOp) and cur in p.values:
            cur = p
            continue
        if isinstance(p, ast.UnaryOp) and cur is p.operand:
            cur = p
            continue
        if isinstance(p, (ast.If, ast.IfExp)) and cur is p.test:
            return True
        if isinstance(p, ast.Assert) and cur is p.test:
            return True
        return False


def _has_literal_ratio(node: ast.AST) -> bool:
    """HOUSE-50/H1 -- True if `node`'s own subtree contains a `Mult`/`Div` `BinOp` with a
    numeric-literal operand on either side: the ">= Nx" shape of a "stood out by at least
    this multiple" claim (`hms > 2.0 * sms`). Structural rather than textual/name-based
    (unlike `_DURATION_RE`) because the two profiled quantities being compared -- one stage's
    EWMA cost against another's -- have no reason to be NAMED `elapsed`/`_ms`/`ratio`/etc.;
    what marks the comparison as magnitude-shaped is the literal multiplier itself, and it
    can be spelled on either side of the operator."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.BinOp) and isinstance(sub.op, (ast.Mult, ast.Div)):
            for side in (sub.left, sub.right):
                if isinstance(side, ast.Constant) and isinstance(side.value, (int, float)) \
                        and not isinstance(side.value, bool):
                    return True
    return False


def _called_names(node: ast.AST) -> set:
    """Every bare-name call target inside `node` -- `foo(...)`, not `obj.foo(...)` (an
    attribute call is out of this scanner's same-file call-graph scope by construction)."""
    names = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
            names.add(sub.func.id)
    return names


def _reachable(start: str, funcs: dict) -> set:
    """FIX-GATE G1 (B4#1): the set of top-level functions `start` can reach through
    same-file bare-name calls, `start` included -- a BFS over the module's own tiny call
    graph, capped implicitly by `funcs`'s size (no external/attribute call is ever
    followed). This is what lets a guard living in a SIBLING helper (`_assert_fast_enough`)
    or a wall-clock reading living in one (`_measure`) still get attributed to the test that
    calls it, instead of only ever looking inside the `test_*` function's own AST subtree."""
    seen = {start}
    stack = [start]
    while stack:
        cur = stack.pop()
        fn = funcs.get(cur)
        if fn is None:
            continue
        for callee in _called_names(fn):
            if callee in funcs and callee not in seen:
                seen.add(callee)
                stack.append(callee)
    return seen


def _seg(source: str, node: ast.AST) -> str:
    try:
        return ast.get_source_segment(source, node) or ""
    except Exception:
        return ""


def scan_source(source: str, filename: str = "<string>") -> list:
    """Every `(func_name, lineno, guard_text)` this scanner flags in one file's source -- a
    top-level `def test_*` NOT decorated `@pytest.mark.timing`/`@pytest.mark.slow`, that can
    REACH (itself, or through same-file bare-name calls, `_reachable`) a wall-clock reading
    AND a guard -- a `Compare`, or a bare `Call` used directly as a boolean test (FIX-GATE
    G1/B4#4: `assert math.isclose(...)` has no `Compare` node at all) -- with a
    duration-shaped operand.

    FIX-GATE G1 widened this past a single function's own AST subtree (B4#1): the guard and
    the wall-clock reading can each live in a DIFFERENT top-level function from the test
    (a shared `_assert_fast_enough(r, elapsed, bound)` helper; a `_measure()` helper that
    times something and hands back only a float) -- `_reachable(test_name, funcs)` computes
    the set of same-file functions the test can reach by a plain `name(...)` call, and both
    conditions (a wall-clock source somewhere in that reachable set; a guard, anywhere in
    it) are now checked over the WHOLE set, not just the test's own body.

    Returns `[("<parse-error>", 0, str(e))]` on a file that does not parse, rather than
    raising -- a scan helper crashing the ratchet on a stray file is worse than one that
    reports the file as its own finding.

    HOUSE-50/H1 widened the wall-clock-SOURCE test past `_WALLCLOCK_SRC_RE` alone: a
    PROFILER-STAGE relative comparison (`test_v031_prof1_per_stage_breakdown`'s
    `hms > 2.0 * sms`) is timed by `tex_runtime/profile.py`'s own `measure()` context
    manager, one module away from the test's own source text, so no literal
    `perf_counter()`/`.time()`/etc. ever appears in this file for `_WALLCLOCK_SRC_RE` to
    find. `has_profiled_cook` (`_COOK_CALL_RE` AND `_PROFILER_READ_RE` both present in the
    reachable set) recognizes that shape instead, and -- ONLY within functions where it
    holds -- additionally accepts a ternary guard (`_is_ternary_guard_node`) and a
    structural ratio (`_has_literal_ratio`) as satisfying the guard/duration-shape tests,
    since a profiled stage's cost variable has no reason to be NAMED `elapsed`/`_ms`/etc."""
    try:
        tree = ast.parse(source, filename)
    except Exception as e:
        return [("<parse-error>", 0, str(e))]

    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    hits = []
    for name, node in funcs.items():
        if not name.startswith("test_"):
            continue
        dec_text = " ".join(_seg(source, d) for d in node.decorator_list)
        if _MARKER_RE.search(dec_text):
            continue
        reachable = _reachable(name, funcs)
        reach_src = " ".join(_seg(source, funcs[n]) for n in reachable)
        has_wallclock = bool(_WALLCLOCK_SRC_RE.search(reach_src))
        has_profiled_cook = bool(_COOK_CALL_RE.search(reach_src) and
                                  _PROFILER_READ_RE.search(reach_src))
        if not (has_wallclock or has_profiled_cook):
            continue
        for fn_name in reachable:
            fn = funcs[fn_name]
            parents = _parent_map(fn)
            for sub in ast.walk(fn):
                if isinstance(sub, ast.Compare):
                    guarded = _is_guard_node(sub, parents) or \
                        (has_profiled_cook and _is_ternary_guard_node(sub, parents))
                    if not guarded:
                        continue
                    operands = [sub.left] + list(sub.comparators)
                elif isinstance(sub, ast.Call):
                    # ATTRIBUTE calls only (`math.isclose(...)`, `self.assertAlmostEqual(...)`,
                    # a `.timeit(...)` bound method): every real B4#4 shape is a named method
                    # on something, never a bare builtin. Excluding a bare-Name call (`any(...)`,
                    # `all(...)`, `len(...)`, `isinstance(...)`) is deliberate, not an oversight:
                    # those are generic aggregation/introspection over an ALREADY-computed
                    # boolean/value, structurally different from a call that IS the magnitude
                    # comparison -- and, concretely, `scan_source`'s own
                    # `if any(_DURATION_RE.search(t) or ... for t in texts):` line would
                    # otherwise self-flag (its generator argument's SOURCE TEXT contains the
                    # substring "duration", case-insensitively, purely because that is this
                    # scanner's own variable name) the moment this file scans itself.
                    if not isinstance(sub.func, ast.Attribute):
                        continue
                    guarded = _is_guard_node(sub, parents) or \
                        (has_profiled_cook and _is_ternary_guard_node(sub, parents))
                    if not guarded:
                        continue
                    operands = list(sub.args) + [kw.value for kw in sub.keywords]
                else:
                    continue
                texts = [_seg(source, o) for o in operands]
                is_profiler_ratio = has_profiled_cook and _has_literal_ratio(sub)
                if any(_DURATION_RE.search(t) or _WALLCLOCK_SRC_RE.search(t) for t in texts) \
                        or is_profiler_ratio:
                    hits.append((name, sub.lineno, _seg(source, sub)))
    return hits


def _tests_relpath(path: str) -> str:
    return "tests/" + os.path.basename(path).replace("\\", "/")


def test_gate47_no_new_wallclock_assertion_without_a_marker(r: SubTestResult):
    print("\n--- GATE-47 (TRK-208): no unmarked wall-clock comparison inside an assertion ---")
    fired = set()
    unallowed = []
    for path in sorted(glob.glob(os.path.join(_TESTS_DIR, "test_*.py"))):
        rel = _tests_relpath(path)
        try:
            source = open(path, encoding="utf-8").read()
        except OSError:
            continue
        for name, lineno, text in scan_source(source, path):
            if name == "<parse-error>":
                unallowed.append(f"{rel}: FAILED TO PARSE -- {text}")
                continue
            key = (rel, name)
            if key in _ALLOWLIST:
                fired.add(key)
                continue
            unallowed.append(f"{rel}::{name} @ {lineno}: `{text}` -- add "
                             f"@pytest.mark.timing (or .slow), or a reviewed _ALLOWLIST "
                             f"entry with a reason")
    stale = sorted(f"{f}::{n}" for f, n in _ALLOWLIST if (f, n) not in fired)
    if unallowed:
        r.fail("GATE-47 wall-clock ratchet",
               f"{len(unallowed)} unmarked, unallowed wall-clock comparison(s):\n  " +
               "\n  ".join(unallowed[:40]))
    elif stale:
        r.fail("GATE-47 wall-clock ratchet (stale allowlist)",
               f"{len(stale)} allowlist entr(y/ies) no longer fire -- the code moved on; "
               f"remove them: " + ", ".join(stale))
    else:
        r.ok(f"{len(fired)} reviewed pre-existing wall-clock comparison(s), 0 unallowed, "
             f"0 stale allowlist entries")


def test_gate47_ratchet_fires_on_the_real_shape_and_stays_green_on_its_neighbours(r: SubTestResult):
    """`test_gate47_no_new_wallclock_assertion_without_a_marker` only proves something if the
    pattern actually fires on the shape it exists for, and only on that shape -- the same
    "is it inert" proof `test_lint1_the_lint_is_not_inert` runs for LINT-1's patterns. Every
    case below is a synthetic snippet, parsed directly (no fixture files, no real allowlist
    entries needed)."""
    print("\n--- GATE-47: the wall-clock scanner fires on the real shape, not its neighbours ---")

    must_red = '''
def test_synthetic_cached_lookup(r):
    import time
    t0 = time.perf_counter()
    do_thing()
    cached_elapsed = time.perf_counter() - t0
    bound = 1.0
    if cached_elapsed <= bound:
        r.ok("fast enough")
    else:
        r.fail("too slow", "nope")
'''
    must_red_boolop = '''
def test_synthetic_boolop_chain(r):
    import time
    t0 = time.perf_counter()
    do_thing()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    if elapsed_ms is not None and elapsed_ms > 0.0 and elapsed_ms < 2000:
        r.ok("within budget")
    else:
        r.fail("out of budget", "nope")
'''
    must_stay_green = {
        "informational only (no comparison at all)": '''
def test_synthetic_benchmark_report(r):
    import time
    t0 = time.perf_counter()
    do_thing()
    elapsed = time.perf_counter() - t0
    r.ok(f"took {elapsed:.3f}s")
''',
        "poll-until-deadline while loop (governs a wait, not pass/fail)": '''
def test_synthetic_poll_loop(r):
    import time
    deadline = time.perf_counter() + 2.0
    result = None
    while result is None and time.perf_counter() < deadline:
        result = try_thing()
    if result is not None:
        r.ok("resolved")
    else:
        r.fail("never resolved", "nope")
''',
        "subprocess/JSON round-trip (measured in a child, immune to this process's tracer)": '''
_PROBE_SRC = "import time, json; t0=time.perf_counter(); print(json.dumps({\\'elapsed\\': time.perf_counter()-t0}))"


def test_synthetic_subprocess_probe(r):
    # The wall-clock reading happens in the CHILD (inside _PROBE_SRC, a module-level
    # constant this function only references by name) -- this function's OWN body never
    # calls a wall-clock source itself, matching `test_v0422_redos.py`'s real
    # ReDoS-guard shape exactly.
    import subprocess, sys, json
    out = subprocess.run([sys.executable, "-c", _PROBE_SRC], capture_output=True, text=True)
    payload = json.loads(out.stdout)
    elapsed = payload["elapsed"]
    if elapsed < 1.0:
        r.ok("child was fast")
    else:
        r.fail("child was slow", "nope")
''',
        "already marked @pytest.mark.timing": '''
@pytest.mark.timing
def test_synthetic_marked(r):
    import time
    t0 = time.perf_counter()
    do_thing()
    elapsed = time.perf_counter() - t0
    if elapsed < 1.0:
        r.ok("fast")
    else:
        r.fail("slow", "nope")
''',
        "duration-shaped name with no wall-clock source in this function at all": '''
def test_synthetic_unrelated_ratio(r):
    compression_ratio = compute_ratio()
    if compression_ratio > 2.0:
        r.ok("good ratio")
    else:
        r.fail("bad ratio", "nope")
''',
    }

    missed = []
    if not scan_source(must_red, "<synthetic>"):
        missed.append("the CI-461 microbenchmark shape (bare `elapsed` names)")
    if not scan_source(must_red_boolop, "<synthetic>"):
        missed.append("a Compare reached through an `and`/`or` chain inside an `if`")

    tripped = [label for label, src in must_stay_green.items() if scan_source(src, "<synthetic>")]

    if missed:
        r.fail("GATE-47 scanner witness (inert)",
               "did not fire on: " + "; ".join(missed))
    elif tripped:
        r.fail("GATE-47 scanner witness (over-tight)",
               "fired on a neighbour that should stay green: " + "; ".join(tripped))
    else:
        r.ok(f"fired on both real shapes, stayed green on {len(must_stay_green)} neighbours")


def test_gate47_g1_closes_four_confirmed_blind_spots(r: SubTestResult):
    """FIX-GATE G1 (B4 findings #1 and #3 and #4): each confirmed blind-spot shape,
    reproduced exactly as the finding described it, must now scan red; a neighbour that
    merely shares some of the same pieces must stay green. (B4#2, the Call-vs-Compare gap
    for `pytest.approx` used the idiomatic way, was already reported NOT a defect -- `assert
    x == pytest.approx(y)` is still a `Compare` and was already caught; nothing to widen
    there.)"""
    print("\n--- GATE-47 (G1): the confirmed scanner blind spots are closed ---")

    must_red = {
        "B4#1a: a guard living in a sibling top-level helper": '''
def _assert_fast_enough(r, elapsed, bound):
    if elapsed <= bound:
        r.ok("fast enough")
    else:
        r.fail("too slow", "nope")


def test_g1_sibling_helper_guard(r):
    import time
    t0 = time.perf_counter()
    do_thing()
    cached_elapsed = time.perf_counter() - t0
    _assert_fast_enough(r, cached_elapsed, 1.0)
''',
        "B4#1b: a helper that times something and hands back only a float": '''
def _measure():
    import time
    t0 = time.perf_counter()
    do_thing()
    return time.perf_counter() - t0


def test_g1_helper_returns_float(r):
    cached_elapsed = _measure()
    if cached_elapsed <= 1.0:
        r.ok("fast enough")
    else:
        r.fail("too slow", "nope")
''',
        "B4#3a: timeit.repeat(...)": '''
def test_g1_timeit_repeat(r):
    import timeit
    times = timeit.repeat(lambda: do_thing(), number=100, repeat=3)
    fastest_ms = min(times) * 1000
    if fastest_ms < 50.0:
        r.ok("fast enough")
    else:
        r.fail("too slow", "nope")
''',
        "B4#3b: Timer(...).timeit(...)": '''
def test_g1_timer_dot_timeit(r):
    import timeit
    t = timeit.Timer(lambda: do_thing())
    elapsed = t.timeit(number=100)
    if elapsed < 1.0:
        r.ok("fast enough")
    else:
        r.fail("too slow", "nope")
''',
        "B4#4: a bare Call (math.isclose) used directly as an assert's test": '''
def test_g1_call_shaped_guard(r):
    import time, math
    t0 = time.perf_counter()
    do_thing()
    cached_elapsed = time.perf_counter() - t0
    bound = 1.0
    assert math.isclose(cached_elapsed, bound, rel_tol=0.2)
    r.ok("close enough")
''',
    }

    must_stay_green = {
        "a sibling helper that is never called (unreachable, not a false negative)": '''
def _unused_helper(r, elapsed, bound):
    if elapsed <= bound:
        r.ok("fast enough")
    else:
        r.fail("too slow", "nope")


def test_g1_neighbour_unused_helper(r):
    result = compute_ratio()
    if result > 2.0:
        r.ok("good ratio")
    else:
        r.fail("bad ratio", "nope")
''',
        "a Call-shaped guard whose arguments are not duration-shaped": '''
def test_g1_neighbour_call_guard_no_duration(r):
    import time, math
    t0 = time.perf_counter()
    do_thing()
    _ = time.perf_counter() - t0
    assert math.isclose(pixel_count(), 2.0, rel_tol=0.1)
    r.ok("close enough")
''',
        "a bare-Name Call (any/all) used as a guard, not an attribute call": '''
def test_g1_neighbour_bare_name_call_guard(r):
    import time
    t0 = time.perf_counter()
    do_thing()
    elapsed_flags = [True, False]
    if any(elapsed_flags):
        r.ok("at least one flag")
    else:
        r.fail("no flags", "nope")
''',
    }

    missed = [label for label, src in must_red.items() if not scan_source(src, "<synthetic>")]
    tripped = [label for label, src in must_stay_green.items() if scan_source(src, "<synthetic>")]
    if missed:
        r.fail("GATE-47 G1 blind spots (still inert)", "did not fire on: " + "; ".join(missed))
    elif tripped:
        r.fail("GATE-47 G1 blind spots (over-tight)",
               "fired on a neighbour that should stay green: " + "; ".join(tripped))
    else:
        r.ok(f"all {len(must_red)} confirmed blind-spot shapes now scan red; "
             f"{len(must_stay_green)} neighbours stay green")


def test_gate47_catches_profiler_stage_relative_comparisons(r: SubTestResult):
    """HOUSE-50/H1: `test_v031_prof1_per_stage_breakdown` (`tests/test_v031_phase2.py`)
    compares two of PROF-1's own per-stage EWMA readings against each other
    (`hms > 2.0 * sms`, inside a ternary `r.ok(...) if ... else r.fail(...)`) and flaked once
    under box load -- a real wall-clock claim this scanner missed on TWO counts at once: the
    `perf_counter()` call lives inside `tex_runtime/profile.py`'s `measure()`, a different
    module `_WALLCLOCK_SRC_RE`'s same-file textual scan cannot see, and the guard is a
    ternary (`ast.IfExp`), which `_is_guard_node` never climbed into. This is the literal
    shape, reproduced in a scratch snippet (never touching the real test file, which is
    fixed separately by marking it `@pytest.mark.timing`)."""
    print("\n--- GATE-47 (HOUSE-50/H1): profiler-stage relative comparisons ---")

    must_red = '''
def test_synthetic_profiler_stage_ratio(r):
    with _armed():
        for _ in range(3):
            tex_engine.cook(term, {"IN": A}, chain_payload=payload, device_mode="cpu")
        snap = P.snapshot()
        stages = {}
        for buckets in snap.values():
            for b in buckets.values():
                if b["stages"]:
                    stages = b["stages"]
        ranked = sorted(stages.items(), key=lambda kv: -kv[1])
        (heavy, hms), (_second, sms) = ranked[0], ranked[1]
        r.ok(f"stage {heavy} stands out") if heavy == "1" and hms > 2.0 * sms else \\
            r.fail("stage ratio", f"did not stand out: {stages}")
'''

    must_stay_green = {
        "profiler seeded with a literal constant, no real cook (pure arithmetic)": '''
def test_synthetic_profiler_seeded_not_measured(r):
    P.record(key, 10.0, 256 * 256)
    got = P.predict(key, 512 * 512)
    r.ok(f"predicts {got:.1f} ms") if got is not None and abs(got - 40.0) < 1e-6 else \\
        r.fail("PROF-1 scale", f"predicted {got!r}, expected 40.0")
''',
        "a real cook, profiler read back, but no ratio -- a plain count/threshold check": '''
def test_synthetic_profiler_cook_but_no_ratio(r):
    with _armed():
        for _ in range(3):
            tex_engine.cook(term, {"IN": A}, chain_payload=payload, device_mode="cpu")
        snap = P.snapshot()
        stages = snap
        if len(stages) < 3:
            r.fail("PROF-1 stages", f"expected a 3-stage breakdown, got {stages}")
        else:
            r.ok(f"a fused chain profiles per stage: {stages}")
''',
        "a real cook and a ratio compare, but never reads the profiler at all": '''
def test_synthetic_cook_ratio_no_profiler_read(r):
    with _armed():
        out = tex_engine.cook(term, {"IN": A}, chain_payload=payload, device_mode="cpu")
        brightness = out.mean().item()
        baseline = 0.4
        r.ok("brighter than baseline") if brightness > 2.0 * baseline else \\
            r.fail("brightness ratio", f"{brightness} vs {baseline}")
''',
    }

    missed = not scan_source(must_red, "<synthetic>")
    tripped = [label for label, src in must_stay_green.items() if scan_source(src, "<synthetic>")]
    if missed:
        r.fail("GATE-47 HOUSE-50/H1 (inert)",
               "did not fire on the profiler-stage relative comparison shape")
    elif tripped:
        r.fail("GATE-47 HOUSE-50/H1 (over-tight)",
               "fired on a neighbour that should stay green: " + "; ".join(tripped))
    else:
        r.ok(f"fired on the profiler-stage relative comparison shape; "
             f"{len(must_stay_green)} neighbours (seeded-not-measured, cook-without-ratio, "
             f"ratio-without-profiler-read) stayed green")


def test_gate47_ratchet_catches_the_v046_ci_failure_shape(r: SubTestResult):
    """The literal proof the brief asks for: restore v0.46.0's own failing test
    (`tests/test_fixobsroute46_r3_torch_global.py::test_r3_microbenchmark_cached_lookup_is_not_slower_than_a_fresh_import`,
    the exact node id CI reported red on every Python version, fixed at `e774d03` /
    `CI-461`) into a SCRATCH file outside the repository, show this scanner reds on it, then
    delete the scratch file -- nothing in the tree is ever touched, so there is nothing to
    revert but the temp file this test itself removes in `finally`."""
    print("\n--- GATE-47: reproduces the v0.46.0 CI failure's own test shape in a scratch "
          "copy, shows it red, and reverts (deletes the scratch copy) ---")
    restored_source = '''
import time
import torch
from TEX_Wrangle import tex_marshalling


def test_r3_microbenchmark_cached_lookup_is_not_slower_than_a_fresh_import(r):
    """Informational-but-gated microbenchmark: repeatedly calling `_torch_mod()` (module-
    global cache, post-warm) must not be slower than a bare `import torch` statement
    executed the same number of times."""
    N = 200_000

    tex_marshalling._torch_mod()   # warm the cache
    t0 = time.perf_counter()
    for _ in range(N):
        tex_marshalling._torch_mod()
    cached_elapsed = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(N):
        import torch as _t   # noqa: F401
    fresh_elapsed = time.perf_counter() - t0

    bound = fresh_elapsed * 2.0 + 1e-6
    if cached_elapsed <= bound:
        r.ok(f"cached _torch_mod(): {cached_elapsed*1e9/N:.1f} ns/call vs bare "
             f"`import torch`: {fresh_elapsed*1e9/N:.1f} ns/call over {N} calls")
    else:
        r.fail("FIX-OBSROUTE R3 microbenchmark", f"cached _torch_mod() ({cached_elapsed:.4f}s) "
               f"was more than 2x slower than a bare `import torch` ({fresh_elapsed:.4f}s) "
               f"over {N} calls -- the cache is not paying for itself")
'''
    fd, path = tempfile.mkstemp(prefix="gate47_restored_v046_r3_", suffix=".py")
    try:
        os.close(fd)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(restored_source)
        with open(path, encoding="utf-8") as fh:
            hits = scan_source(fh.read(), path)
        if hits and hits[0][0] == "test_r3_microbenchmark_cached_lookup_is_not_slower_than_a_fresh_import":
            r.ok(f"restored v0.46.0 shape scanned red, as CI's coverage-on run originally "
                 f"was: {hits[0]}")
        else:
            r.fail("GATE-47 v0.46.0 repro", f"expected the restored shape to scan red; got {hits}")
    finally:
        try:
            os.remove(path)   # revert: the scratch copy never persists
        except OSError:
            pass
