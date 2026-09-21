"""
LNT-2 — ban the exactness claim that is not portable (a tribal invariant that has
already bitten three releases in a row).

Every local gate can be green and the Linux CI lane still red, because the local
gates run on Windows. Three consecutive releases were tagged, pushed and refused
for the SAME defect wearing three different faces: **a test pinning bit-exactness
on something whose contract was never bit-exact.**

  1. A test pinned a SHA-256 of the bytes a product writer wrote through a TEXT-mode
     handle. Text mode emits the platform's newline, so the hash was CRLF here and LF
     on the runner and an unchanged file read as drifted.
  2. A test compared an EAGER noise body against the same noise called through the
     three-tier compilation cache (eager -> jit.trace -> torch.compile) and asserted
     bit-equality. The runner's fuser reassociates; it failed by 2**-24, one fp32 ulp.
  3. A second row of the same class: two DISPATCHED calls compared bit-exactly, with
     the cache's once-per-process callable swap free to land between them.

This lint scans test sources for both shapes, so the lesson is machine-enforced rather
than doc-only — and because it is a test it runs on the Linux lane automatically and
locally before a push, which is the whole point.

MEASURED BEFORE CHOSEN, because a rule that fires 400 times gets switched off and
protects nobody. The naive grep (`torch.equal|== 0.0|assert .*== <n>`) returns 429 hits
across 63 files here, and almost all are legitimate -- integers, shapes, discrete ids,
two results of the SAME path. The census that mattered, over all 97 test sources:

  * 907 `assert`ed equalities in total;
  * 239 of those are FLOAT-exact (152 `torch.equal`/`_bitwise_same`, 87 `==` against a
    float literal) -- the rest compare ints, shapes, strings, sets, error codes;
  * 4 of the 239 have an operand reaching a tiered dispatcher. That is the rule below,
    and 4 is the measured false-positive count on the tree it was written against --
    all 4 turned out legitimate and each now carries a reasoned marker, so the steady
    state is ZERO;
  * 14 `!=`-against-a-float and 37 `not torch.equal(...)` sites, none tiered, and
    an inequality is robust in the direction a ulp moves anyway;
  * 4 `hashlib` sites in tests/, of which 1 hashes file bytes -- already normalised.
    Rule B's false-positive count is 0.

  RULE A -- TIERED EXACTNESS. A `torch.equal(...)` / `_bitwise_same(...)` call, or an
  `==` against a float literal, whose operand reaches a TIERED dispatcher. "Is this an
  exact comparison" is not the signal -- exact comparisons are overwhelmingly fine
  (shapes, ids, integers, two results of one path). "Does an operand come from a tier
  that can be swapped under it" is. The tiered set is DERIVED from the product source
  (`_derive_tiered_entry_points`), never hardcoded, the way `tex_memory._non_local_fns()`
  derives from the stdlib registry -- a hardcoded list rots the moment someone adds a
  tiered builtin.

  RULE B -- TEXT-MODE BYTE HASH. `hashlib.<algo>(x)` in a test where `x` reaches a raw
  binary file read with no intervening newline normalisation. Raw file bytes carry the
  writer's newline convention, so hashing them pins a platform, not a payload.

ESCAPE HATCH. A legitimate exact comparison declares itself with an inline
`# lnt2-ok: <reason>` on the statement or in the comment block directly above it. The
reason is MANDATORY and is itself linted: a bare marker, or one with a reason too short
to say anything, fails this test. A suppression with no reason is how a lint becomes
noise, and noise is how a lint gets switched off.

Stdlib only, no numpy (invariant 1), no torch. The lint READS sources -- it never
imports the modules it lints, so it costs no import time and cannot be defeated by an
import-time side effect.
"""
import ast
import os
import re

from helpers import SubTestResult

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # package root
_TESTS = os.path.join(_PKG, "tests")
_SELF = os.path.basename(__file__)

_MARKER = "lnt2-ok"
_MARKER_RE = re.compile(r"#\s*" + _MARKER + r"\b\s*:?(?P<reason>.*)$")
_MIN_REASON = 12   # characters; short enough for a terse reason, long enough to be one


# ── the derived tiered set ───────────────────────────────────────────────────

def _parse_file(path):
    with open(path, encoding="utf-8") as f:
        return ast.parse(f.read(), filename=path)


def _tail_name(func):
    """`f(...)` -> 'f'; `a.b.f(...)` -> 'f'. The dotted head is deliberately ignored:
    a test reaches these through `noise._worley2d`, `TEXStdlib.fn_voronoi` or a bare
    import, and the tail is the part that identifies the callable in all three."""
    return func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)


def _referenced_names(node):
    """Every name `node` calls OR mentions. Mentions count: handing a function object
    to a higher-order helper (`_alligator_nd(_worley2d, ...)`) is a deferred call, and
    it carries the tier exactly as a direct call does. Missing that edge is how a
    hand-kept list would have quietly dropped `alligator`."""
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
    return out


def _functions_of(tree):
    return {n.name: n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _closure(funcs, seed):
    """Every function in `funcs` that transitively reaches `seed`."""
    reached = set(seed)
    changed = True
    while changed:
        changed = False
        for name, node in funcs.items():
            if name not in reached and _referenced_names(node) & reached:
                reached.add(name)
                changed = True
    return reached


def _derive_tiered_entry_points(pkg=_PKG):
    """DERIVE, from product source, every name whose result can come from a swappable
    compilation tier. Three layers, each read off the code rather than listed here:

      1. the runtime module's module-level tier caches -- any name bound to a
         `_TieredCache(...)`, whatever it is called;
      2. the functions that dispatch through one (`<cache>.call(...)`), plus the
         transitive closure of everything in that module that reaches them;
      3. the stdlib `fn_*` impls that reach layer 2, plus the LANGUAGE names their
         `@stdlib(...)` decorator registers -- so a comparison whose operand is a
         cooked program SOURCE STRING is caught too.

    Returns (caches, runtime_fns, stdlib_fns, language_names). Raises if layer 1 comes
    back empty: a silent empty set would turn this lint into a no-op, which is worse
    than a loud failure.
    """
    noise_tree = _parse_file(os.path.join(pkg, "tex_runtime", "noise.py"))

    caches = set()
    for n in noise_tree.body:
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call):
            if _tail_name(n.value.func) == "_TieredCache":
                caches |= {t.id for t in n.targets if isinstance(t, ast.Name)}
    if not caches:
        raise AssertionError(
            "LNT-2 could not find any _TieredCache instance in tex_runtime/noise.py — "
            "the tier mechanism was renamed or moved. Re-derive it here; do NOT let "
            "this lint degrade to a no-op.")

    noise_funcs = _functions_of(noise_tree)
    seeds = set()
    for name, node in noise_funcs.items():
        for c in ast.walk(node):
            if (isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                    and c.func.attr == "call"
                    and isinstance(c.func.value, ast.Name)
                    and c.func.value.id in caches):
                seeds.add(name)
    if not seeds:
        raise AssertionError(
            "LNT-2 found tier caches but no dispatch site (`<cache>.call(...)`) in "
            "tex_runtime/noise.py — the dispatch entry point was renamed. Re-derive it.")
    runtime_fns = _closure(noise_funcs, seeds)

    # LIB-1: the `fn_*` impls (and their private helpers) no longer live in `stdlib.py`
    # itself — that file is now the facade over seven per-domain leaves plus the shared
    # substrate `stdlib_core.py`. Parse all of them and merge, or this derivation silently
    # empties out (the exact failure mode the docstring above says is worse than a loud one).
    std_funcs = {}
    for _fname in ("stdlib.py", "stdlib_core.py", "stdlib_math.py", "stdlib_color.py",
                   "stdlib_sample.py", "stdlib_noise.py", "stdlib_sdf.py",
                   "stdlib_string.py", "stdlib_array.py"):
        std_funcs.update(_functions_of(_parse_file(os.path.join(pkg, "tex_runtime", _fname))))
    stdlib_fns = _closure(std_funcs, runtime_fns) - runtime_fns

    language = set()
    for name in stdlib_fns:
        for d in std_funcs[name].decorator_list:
            if isinstance(d, ast.Call) and _tail_name(d.func) == "stdlib":
                if d.args and isinstance(d.args[0], ast.Constant):
                    language.add(d.args[0].value)
                for kw in d.keywords:
                    if kw.arg == "aliases" and isinstance(kw.value, (ast.Tuple, ast.List)):
                        language |= {e.value for e in kw.value.elts
                                     if isinstance(e, ast.Constant)}
    return caches, runtime_fns, stdlib_fns, language


# ── the scanner ──────────────────────────────────────────────────────────────

_EXACT_TENSOR_CALLS = {"equal", "_bitwise_same"}
_HASHERS = {"sha256", "sha1", "sha512", "md5", "blake2b", "blake2s", "new"}
_NORMALISERS = {"replace", "splitlines", "decode", "translate"}
_BINARY_READS = {"read_bytes"}


def _statements(node):
    """Every statement inside `node`, innermost included."""
    for n in ast.walk(node):
        if isinstance(n, ast.stmt):
            yield n


def _own_expressions(stmt):
    """The expression subtrees belonging to `stmt` ITSELF — its test, its value, its
    call args — never a nested statement's. Without this split every finding would be
    attributed to the outermost `def`, and one marker on that `def` would silence the
    whole function. A suppression must scope to the line it excuses."""
    for _field, value in ast.iter_fields(stmt):
        for item in (value if isinstance(value, list) else [value]):
            if isinstance(item, ast.expr):
                yield item


class _Scan:
    """One test source, scanned. Never imports anything it reads."""

    def __init__(self, filename, src, tiered_names, language):
        self.filename = filename
        self.lines = src.splitlines()
        self.tiered = tiered_names           # runtime_fns | stdlib_fns
        self.language = language
        self.findings = []                   # actionable hits
        self.marker_problems = []            # bad `# lnt2-ok` markers
        self.suppressed = []                 # hits carrying a good marker
        self._seen = set()                   # (line, rule) already recorded
        self.tree = ast.parse(src, filename=filename)

    # -- marker lookup -----------------------------------------------------
    def _marker_lines(self, node):
        """Where a suppression for `node` may live: the flagged EXPRESSION's own lines
        (so a trailing `# lnt2-ok:` works, including on an `elif` header) plus the
        contiguous comment block directly above it (so the usual placement works). The
        span is deliberately the expression's, not the enclosing block's — a marker must
        not reach past the comparison it excuses."""
        lo = getattr(node, "lineno", 1)
        hi = getattr(node, "end_lineno", lo) or lo
        idxs = list(range(lo, hi + 1))
        i = lo - 1
        while i >= 1 and self.lines[i - 1].lstrip().startswith("#"):
            idxs.append(i)
            i -= 1
        return idxs

    def _suppression(self, stmt):
        """(suppressed?, reason_or_problem). A marker with no usable reason is NOT a
        suppression — it is reported as a marker problem, so the bare form cannot be
        used to silence anything."""
        for i in self._marker_lines(stmt):
            if i - 1 >= len(self.lines):
                continue
            m = _MARKER_RE.search(self.lines[i - 1])
            if not m:
                continue
            reason = m.group("reason").strip().strip('"').strip("'").strip()
            if len(reason) < _MIN_REASON:
                return False, ("bare", i, reason)
            return True, ("ok", i, reason)
        return False, None

    # -- taint -------------------------------------------------------------
    def _walk_function(self, fn):
        tiered_vars = {}
        byte_vars = {}
        bin_handles = set()

        for n in ast.walk(fn):
            if isinstance(n, ast.With):
                for item in n.items:
                    c = item.context_expr
                    if not (isinstance(c, ast.Call) and _tail_name(c.func) == "open"):
                        continue
                    mode = None
                    if len(c.args) > 1 and isinstance(c.args[1], ast.Constant):
                        mode = c.args[1].value
                    for kw in c.keywords:
                        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                            mode = kw.value.value
                    if (isinstance(mode, str) and "b" in mode
                            and isinstance(item.optional_vars, ast.Name)):
                        bin_handles.add(item.optional_vars.id)

        def tier_origin(expr):
            """The tiered name `expr` reaches, or None. Three origins, any depth:
            a call to a tiered function, a local already carrying one, or a string
            literal that is a PROGRAM calling a tiered language builtin."""
            for n in ast.walk(expr):
                if isinstance(n, ast.Call):
                    nm = _tail_name(n.func)
                    if nm in self.tiered:
                        return nm
                if isinstance(n, ast.Name) and n.id in tiered_vars:
                    return tiered_vars[n.id]
                if isinstance(n, ast.Constant) and isinstance(n.value, str):
                    if len(n.value) <= 8192:
                        for b in self.language:
                            if re.search(r"\b" + re.escape(b) + r"\s*\(", n.value):
                                return b + "() in a cooked program"
            return None

        def normalised(expr):
            return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                       and n.func.attr in _NORMALISERS for n in ast.walk(expr))

        def byte_origin(expr):
            for n in ast.walk(expr):
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                    if n.func.attr in _BINARY_READS:
                        return "Path.read_bytes()"
                    if (n.func.attr == "read" and isinstance(n.func.value, ast.Name)
                            and n.func.value.id in bin_handles):
                        return "open(..., 'rb').read()"
                if isinstance(n, ast.Name) and n.id in byte_vars:
                    return byte_vars[n.id]
            return None

        # one forward pass binding locals (source order is what a reader sees, and the
        # rule over-approximates on purpose: a name rebound later still reads tainted)
        for stmt in _statements(fn):
            if not isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                continue
            value = stmt.value
            if value is None:
                continue
            targets = (stmt.targets if isinstance(stmt, ast.Assign)
                       else [stmt.target])
            t = tier_origin(value)
            b = None if normalised(value) else byte_origin(value)
            for tg in targets:
                for n in ast.walk(tg):
                    if isinstance(n, ast.Name):
                        if t:
                            tiered_vars[n.id] = t
                        if b:
                            byte_vars[n.id] = b

        for stmt in _statements(fn):
            self._check_stmt(fn, stmt, tier_origin, byte_origin, normalised)

    def _check_stmt(self, fn, stmt, tier_origin, byte_origin, normalised):
        for expr in _own_expressions(stmt):
            for n in ast.walk(expr):
                self._check_node(fn, n, tier_origin, byte_origin, normalised)

    def _check_node(self, fn, n, tier_origin, byte_origin, normalised):
        hit = (self._rule_a(n, tier_origin) or
               self._rule_b(n, byte_origin, normalised))
        if not hit:
            return
        rule, detail = hit
        key = (n.lineno, n.col_offset, rule)
        if key in self._seen:
            return          # a nested def's body is reached by its own walk as well
        self._seen.add(key)
        supp, info = self._suppression(n)
        record = {"file": self.filename, "line": n.lineno, "fn": fn.name,
                  "rule": rule, "detail": detail, "src": self._src(n.lineno)}
        if info and info[0] == "bare":
            self.marker_problems.append(
                f"{self.filename}:{info[1]}: `# {_MARKER}` with no usable reason "
                f"(got {info[2]!r}; need >= {_MIN_REASON} characters saying WHY the "
                f"comparison is portable). A bare suppression is not accepted.")
            self.findings.append(record)
        elif supp:
            record["reason"] = info[2]
            self.suppressed.append(record)
        else:
            self.findings.append(record)

    def _rule_a(self, n, tier_origin):
        ops = None
        if isinstance(n, ast.Call) and _tail_name(n.func) in _EXACT_TENSOR_CALLS:
            if _tail_name(n.func) == "equal" and not (
                    isinstance(n.func, ast.Attribute)
                    and _tail_name(n.func.value) == "torch"):
                return None
            ops = list(n.args)
            shape = "bit-equality call"
        elif (isinstance(n, ast.Compare) and len(n.ops) == 1
              and isinstance(n.ops[0], ast.Eq)):
            sides = [n.left, n.comparators[0]]
            if not any(isinstance(s, ast.Constant) and isinstance(s.value, float)
                       for s in sides):
                return None
            ops = sides
            shape = "== against a float literal"
        if ops is None:
            return None
        origins = [o for o in (tier_origin(x) for x in ops) if o]
        if not origins:
            return None
        return ("A", f"{shape}; operand reaches the tiered dispatcher via "
                     f"{sorted(set(origins))}")

    def _rule_b(self, n, byte_origin, normalised):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr in _HASHERS
                and _tail_name(n.func.value) == "hashlib"):
            return None
        args = [a for a in n.args if not isinstance(a, ast.Constant)] or n.args
        for a in args:
            if normalised(a):
                continue
            origin = byte_origin(a)
            if origin:
                return ("B", f"hashes raw file bytes from {origin} with no newline "
                             f"normalisation — the hash pins the writer's platform")
        return None

    def _src(self, lineno):
        return self.lines[lineno - 1].strip()[:140] if lineno - 1 < len(self.lines) else ""

    def run(self):
        for n in ast.walk(self.tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._walk_function(n)
        return self


def _scan_source(src, filename, tiered_names, language):
    return _Scan(filename, src, tiered_names, language).run()


def _bad_markers_in(src, filename):
    """Every `lnt2-ok` in `src` that is not a well-formed, reasoned suppression."""
    out = []
    for i, line in enumerate(src.splitlines(), 1):
        if _MARKER not in line:
            continue
        m = _MARKER_RE.search(line)
        reason = m.group("reason").strip() if m else ""
        if not m or len(reason) < _MIN_REASON:
            out.append(f"{filename}:{i}: {line.strip()[:110]}")
    return out


# ── the tests ────────────────────────────────────────────────────────────────

def _test_sources():
    for fn in sorted(os.listdir(_TESTS)):
        if not fn.endswith(".py") or fn == _SELF:
            continue
        path = os.path.join(_TESTS, fn)
        try:
            with open(path, encoding="utf-8") as f:
                yield fn, f.read()
        except (OSError, UnicodeDecodeError):
            continue


def test_no_unportable_equality_ban(r: SubTestResult):
    print("\n--- LNT-2: unportable-exactness lint ---")

    try:
        caches, runtime_fns, stdlib_fns, language = _derive_tiered_entry_points()
    except Exception as e:
        r.fail("LNT-2 tiered-set derivation", f"{type(e).__name__}: {e}")
        return
    tiered = runtime_fns | stdlib_fns
    r.ok(f"tiered set DERIVED from product source: {len(caches)} tier cache(s), "
         f"{len(runtime_fns)} runtime fn(s) {sorted(runtime_fns)}, "
         f"{len(stdlib_fns)} stdlib fn(s), "
         f"{len(language)} language name(s) {sorted(language)}")

    findings, suppressed, marker_problems, scanned = [], [], [], 0
    for fn, src in _test_sources():
        try:
            s = _scan_source(src, fn, tiered, language)
        except SyntaxError:
            continue
        scanned += 1
        findings += s.findings
        suppressed += s.suppressed
        # every `lnt2-ok` in the file, well-formed or not -- so a bare marker is caught
        # even where it silences nothing, and a stale one cannot rot into a habit.
        marker_problems += _bad_markers_in(src, fn)

    marker_problems = sorted(set(marker_problems))

    rule_a = [f for f in findings if f["rule"] == "A"]
    rule_b = [f for f in findings if f["rule"] == "B"]

    if rule_a:
        r.fail("LNT-2 rule A (tiered exactness)",
               f"{len(rule_a)} exact comparison(s) on a value from a swappable "
               f"compilation tier — green here, one fp32 ulp red on another "
               f"toolchain. Compare like with like, settle the tier first, or band "
               f"it; if it really is portable, say so with `# {_MARKER}: <reason>`:\n  "
               + "\n  ".join(f"{f['file']}:{f['line']} ({f['fn']}) {f['detail']}\n"
                             f"      {f['src']}" for f in rule_a[:20]))
    else:
        r.ok(f"rule A: no unguarded exact comparison on a tiered value "
             f"({scanned} test sources scanned)")

    if rule_b:
        r.fail("LNT-2 rule B (text-mode byte hash)",
               f"{len(rule_b)} hash(es) of raw file bytes with no newline "
               f"normalisation — the pin records this platform's newline, not the "
               f"payload. Normalise (`.replace(b'\\r\\n', b'\\n')`) before hashing:\n  "
               + "\n  ".join(f"{f['file']}:{f['line']} ({f['fn']}) {f['detail']}\n"
                             f"      {f['src']}" for f in rule_b[:20]))
    else:
        r.ok("rule B: no unnormalised hash of raw file bytes in tests/")

    if marker_problems:
        r.fail(f"LNT-2 `{_MARKER}` hygiene",
               f"a suppression must carry a REASON — {len(marker_problems)} does not:\n  "
               + "\n  ".join(marker_problems[:20]))
    else:
        r.ok(f"every `# {_MARKER}` suppression carries a reason "
             f"({len(suppressed)} in the tree)")


# ── acceptance: the three instances this lint exists for ─────────────────────
#
# Each pair below is the assertion as it was WHEN IT BROKE CI, and as it stands after
# the fix. The lint has to fire on the first and be silent on the second — a lint that
# cannot demonstrate that on its own founding defects is a decoration. The snippets are
# transcribed from the fix commits, trimmed to the statements that matter.

_I1_BEFORE = '''
def _hash_manifest(m) -> dict:
    d = m.to_dict()
    path = tex_tool.write_tool(m, tempfile.mkdtemp())
    with open(path, "rb") as fh:
        written = fh.read()
    return {
        "to_dict": hashlib.sha256(json.dumps(d, sort_keys=True).encode("utf-8")).hexdigest(),
        "written_bytes": hashlib.sha256(written).hexdigest(),
    }
'''

_I1_AFTER = '''
def _hash_manifest(m) -> dict:
    d = m.to_dict()
    path = tex_tool.write_tool(m, tempfile.mkdtemp())
    with open(path, "rb") as fh:
        written = fh.read()
    return {
        "to_dict": hashlib.sha256(json.dumps(d, sort_keys=True).encode("utf-8")).hexdigest(),
        "written_bytes": hashlib.sha256(written.replace(b"\\r\\n", b"\\n")).hexdigest(),
    }
'''

_I2_BEFORE = '''
def test_ask5_matches_worley_f1_winner(r):
    from TEX_Wrangle.tex_runtime import noise
    dx_off, dy_off = noise._get_worley_offsets(x.device, x.dim())
    f1_direct = noise._worley2d_f1(x, y, dx_off, dy_off)
    f1_via_dispatch = noise._worley2d(x, y, return_f2=False)
    md = (f1_direct - f1_via_dispatch).abs().max().item()
    assert md == 0.0, f"_worley2d_f1 direct vs dispatched maxdiff {md}"
'''

_I2_AFTER = '''
def test_ask5_matches_worley_f1_winner(r):
    from TEX_Wrangle.tex_runtime import noise
    dx_off, dy_off = noise._get_worley_offsets(x.device, x.dim())
    dist = noise._worley2d_core(x, y, dx_off, dy_off)
    winner = dist.min(dim=0).indices
    at_winner = torch.sqrt(torch.gather(dist, 0, winner.unsqueeze(0)).squeeze(0))
    f1_direct = noise._worley2d_f1(x, y, dx_off, dy_off)
    assert torch.equal(f1_direct, at_winner), "the id names a cell f1 does not measure"
'''

_I3_BEFORE = '''
def test_ask5_voronoi_unchanged(r):
    vor = TEXStdlib.fn_voronoi(u, v)
    f1 = TEXStdlib.fn_worley_f1(u, v)
    assert torch.equal(vor, f1), "voronoi no longer bit-identical to worley_f1"
    code_v = "@OUT = vec4(vec3(voronoi(u * 8.0, v * 8.0)), 1.0);"
    code_f1 = "@OUT = vec4(vec3(worley_f1(u * 8.0, v * 8.0)), 1.0);"
    for tier in ("interp", "codegen"):
        md = max_diff(run_tier(code_v, {}, tier), run_tier(code_f1, {}, tier))
        assert md == 0.0, f"[{tier}] whole-program voronoi vs worley_f1 maxdiff {md}"
'''

_I3_AFTER = '''
def test_ask5_voronoi_unchanged(r):
    _settle_worley_tier(u, v)
    vor = TEXStdlib.fn_voronoi(u, v)
    f1 = TEXStdlib.fn_worley_f1(u, v)
    # lnt2-ok: both sides settled to the same tier before comparing (_settle_worley_tier)
    assert torch.equal(vor, f1), "voronoi no longer bit-identical to worley_f1"
    code_v = "@OUT = vec4(vec3(voronoi(u * 8.0, v * 8.0)), 1.0);"
    code_f1 = "@OUT = vec4(vec3(worley_f1(u * 8.0, v * 8.0)), 1.0);"
    for tier in ("interp", "codegen"):
        md = max_diff(run_tier(code_v, {}, tier), run_tier(code_f1, {}, tier))
        # lnt2-ok: both sides settled to the same tier before comparing (_settle_worley_tier)
        assert md == 0.0, f"[{tier}] whole-program voronoi vs worley_f1 maxdiff {md}"
'''

# A bare marker must not silence anything: the reason is the point.
_BARE_MARKER = '''
def test_bare(r):
    vor = TEXStdlib.fn_voronoi(u, v)
    f1 = TEXStdlib.fn_worley_f1(u, v)
    assert torch.equal(vor, f1)  # lnt2-ok
'''

_EMPTY_REASON = '''
def test_empty(r):
    vor = TEXStdlib.fn_voronoi(u, v)
    f1 = TEXStdlib.fn_worley_f1(u, v)
    assert torch.equal(vor, f1)  # lnt2-ok: ok
'''


def test_lnt2_catches_the_three_historical_instances(r: SubTestResult):
    """The acceptance test. Each of the three defects that blocked a release must FIRE,
    and each fixed form must be SILENT — otherwise this lint is decoration."""
    print("\n--- LNT-2: acceptance — the three instances that blocked three releases ---")
    try:
        _, runtime_fns, stdlib_fns, language = _derive_tiered_entry_points()
    except Exception as e:
        r.fail("LNT-2 acceptance setup", f"{type(e).__name__}: {e}")
        return
    tiered = runtime_fns | stdlib_fns

    def hits(src, rule=None):
        s = _scan_source(src, "<acceptance>", tiered, language)
        f = s.findings
        return [x for x in f if rule is None or x["rule"] == rule]

    cases = (
        ("instance 1 — SHA-256 of bytes a product writer wrote in TEXT mode "
         "(CRLF here, LF on the runner)", _I1_BEFORE, _I1_AFTER, "B"),
        ("instance 2 — eager noise body vs the same noise through the tier cache, "
         "asserted bit-equal (failed by 2**-24)", _I2_BEFORE, _I2_AFTER, "A"),
        ("instance 3 — two DISPATCHED calls compared bit-exactly, with the "
         "once-per-process tier swap free to land between them", _I3_BEFORE, _I3_AFTER, "A"),
    )
    for label, before, after, rule in cases:
        try:
            fired = hits(before, rule)
            assert fired, "the lint did NOT fire on the pre-fix form"
            silent = hits(after)
            assert not silent, ("the lint fires on the FIXED form: " +
                                "; ".join(f"{h['line']}:{h['detail']}" for h in silent))
            r.ok(f"rule {rule} fires on the pre-fix form ({len(fired)} hit"
                 f"{'s' if len(fired) > 1 else ''}) and is silent on the fix — {label}")
        except AssertionError as e:
            r.fail(f"LNT-2 acceptance ({label})", str(e))

    # and the escape hatch cannot be abused
    for label, src in (("a bare `# lnt2-ok`", _BARE_MARKER),
                       ("a reason too short to say anything", _EMPTY_REASON)):
        try:
            s = _scan_source(src, "<acceptance>", tiered, language)
            assert s.marker_problems, f"{label} was accepted as a suppression"
            assert s.findings, f"{label} silenced the finding"
            assert not s.suppressed, f"{label} counted as a valid suppression"
            r.ok(f"{label} is rejected — the finding stands AND the marker is reported")
        except AssertionError as e:
            r.fail("LNT-2 escape-hatch hygiene", str(e))


if __name__ == "__main__":
    _r = SubTestResult()
    test_no_unportable_equality_ban(_r)
    test_lnt2_catches_the_three_historical_instances(_r)
    _r.summary()
