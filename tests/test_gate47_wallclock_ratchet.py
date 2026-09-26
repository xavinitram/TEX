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
This is the ratchet CI-461's hand-back asked for.

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
_WALLCLOCK_SRC_RE = re.compile(
    r"perf_counter\(\)|time\.time\(\)|timeit\.default_timer\(\)|timeit\.timeit\(|"
    r"\.monotonic\(\)|\.process_time\(\)"
)

_MARKER_RE = re.compile(r"mark\.timing|mark\.slow")

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


def _is_guard_compare(node: ast.Compare, parents: dict) -> bool:
    """True when `node` is (or sits under, through any chain of `and`/`or`/`not`) the `test`
    of an `if` or an `assert` -- the two shapes that decide pass/fail. A `while`/`for` test
    walks up to a node type this never returns True for, so a poll-until-deadline loop is
    excluded by construction, not by a special case."""
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


def _seg(source: str, node: ast.AST) -> str:
    try:
        return ast.get_source_segment(source, node) or ""
    except Exception:
        return ""


def scan_source(source: str, filename: str = "<string>") -> list:
    """Every `(func_name, lineno, compare_text)` this scanner flags in one file's source --
    a top-level `def test_*` NOT decorated `@pytest.mark.timing`/`@pytest.mark.slow`, whose
    own body reads a wall-clock value AND contains a guard `Compare` (see `_is_guard_compare`)
    with a duration-shaped operand. Returns `[("<parse-error>", 0, str(e))]` on a file that
    does not parse, rather than raising -- a scan helper crashing the ratchet on a stray file
    is worse than one that reports the file as its own finding."""
    try:
        tree = ast.parse(source, filename)
    except Exception as e:
        return [("<parse-error>", 0, str(e))]
    parents = _parent_map(tree)
    hits = []
    for node in tree.body:
        if not (isinstance(node, ast.FunctionDef) and node.name.startswith("test_")):
            continue
        dec_text = " ".join(_seg(source, d) for d in node.decorator_list)
        if _MARKER_RE.search(dec_text):
            continue
        func_text = _seg(source, node)
        if not _WALLCLOCK_SRC_RE.search(func_text):
            continue
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Compare) or not _is_guard_compare(sub, parents):
                continue
            operands = [sub.left] + list(sub.comparators)
            texts = [_seg(source, o) for o in operands]
            if any(_DURATION_RE.search(t) or _WALLCLOCK_SRC_RE.search(t) for t in texts):
                hits.append((node.name, sub.lineno, _seg(source, sub)))
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
