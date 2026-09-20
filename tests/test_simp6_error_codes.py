"""SIMP-6 — every error code the product can emit is a code some test names.

An `E####`/`W####` is the public contract between TEX and whatever draws its diagnostics:
an editor gutter, a host's error panel, a CLI. A code nobody tests is a code the next
change is free to re-spell, re-severity or quietly stop emitting, and the only reader who
notices is the one whose editor stopped underlining. A census of the tree when this landed
found **88** distinct codes and **46** of them named by no test at all; the rows in
`tests/test_simp6_error_code_rows.py` retired thirty of those, and the sixteen that
remain are listed beside the pin with the reason each has no trigger.

This file holds three derivations, none of them a typed list of codes:

1. **The ratchet** — count the codes no test names, pin the count, let it move DOWN only.
   A code that gains a test lowers the pin (the row says so, by name); a NEW code that
   arrives without a test pushes the count above the pin and reds, naming it.
2. **Family declarations** — one code raised from several positions with several messages
   is a family, not a duplicate: `E2010` is "missing semicolon" from sixteen parser
   positions. Nothing in the tree recorded which codes are families, so a reviewer had no
   way to tell a legitimate family from a code that was copy-pasted. They are declared
   here, each with the reason it is one, and the declaration is checked against the tree.
3. **Cross-file agreement** — three codes are emitted from more than one MODULE with
   different message text. Those are the ones that can drift into two different editor
   experiences under one code, so each gets a row asserting the sites agree on severity
   and on hint shape.

The population for (1) is every `E####`/`W####` token the product's own source names —
deliberately WIDER than "every site that constructs a diagnostic". A code named only in a
constructor default or in a comment describing a contract is still a code the product
knows and a host may receive, and a wider population cannot be shrunk by moving a code
into a different syntactic position. (2) and (3) need the construction itself, so they
read the AST instead.

A code counts as TESTED when a file under `tests/` names it, which is wide in the same
direction: of the 42 codes that counted as tested when this landed, 27 were named only
inside a comment or a docstring. Tightening that to "named in code" would be a truer
ratchet over a much larger backlog (73 untested rather than 46) and is a separate
decision; the rule here is the blunt one. Its cost is that naming a code in a comment
under `tests/` retires it without testing anything. Do not.

No product code is imported here: the file reads the tree as text. It needs no host, no
CUDA and no compiler.
"""
import ast
import collections
import os
import re

from helpers import SubTestResult

# ── The tree, as data ─────────────────────────────────────────────────

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.dirname(_TESTS_DIR)
_THIS_FILE = os.path.basename(__file__)

# Everything that is not product code. A code named in a test, a benchmark or a tool is
# not the product emitting it.
_NOT_PRODUCT = {
    "tests", "benchmarks", "tools", "docs", "examples", "editor_build", "assets",
    "results", "results_test", "stock", "wiki", "js",
    ".git", ".github", ".tex_cache", "__pycache__",
}

_CODE = re.compile(r"\b([EW]\d{4})\b")


def _product_files():
    """(absolute path, slash-separated path relative to the package) for every product
    `.py` file. Slash-separated so a failure message reads the same on either OS."""
    out = []
    for root, dirs, files in os.walk(_PKG_DIR):
        rel = os.path.relpath(root, _PKG_DIR)
        top = "" if rel == "." else rel.split(os.sep)[0]
        if top in _NOT_PRODUCT:
            dirs[:] = []
            continue
        for fn in sorted(files):
            if fn.endswith(".py"):
                p = os.path.join(root, fn)
                out.append((p, os.path.relpath(p, _PKG_DIR).replace(os.sep, "/")))
    return sorted(out, key=lambda t: t[1])


def _codes_named_by_product():
    """code -> sorted list of "file:line" where the product's source names it."""
    seen = collections.defaultdict(list)
    for path, rel in _product_files():
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                for code in sorted(set(_CODE.findall(line))):
                    seen[code].append(f"{rel}:{i}")
    return seen


def _codes_named_by_tests():
    """code -> set of test file names that name it.

    THIS file is excluded from the population on purpose. The pinned backlog below spells
    out the codes it counts, and the family and cross-file rows have to spell four more to
    declare them; both are bookkeeping, not coverage — counting them would let the ratchet
    be moved by writing a comment.
    """
    hits = collections.defaultdict(set)
    for fn in sorted(os.listdir(_TESTS_DIR)):
        if not fn.endswith(".py") or fn == _THIS_FILE:
            continue
        with open(os.path.join(_TESTS_DIR, fn), encoding="utf-8") as fh:
            src = fh.read()
        for code in set(_CODE.findall(src)):
            hits[code].add(fn)
    return hits


# ── Diagnostic construction sites, from the AST ───────────────────────

_Site = collections.namedtuple("_Site", "file line callee message hint severity")


def _const_str(node):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _callee_name(node):
    f = node.func
    parts = []
    while isinstance(f, ast.Attribute):
        parts.append(f.attr)
        f = f.value
    if isinstance(f, ast.Name):
        parts.append(f.id)
    return ".".join(reversed(parts)) or "<call>"


def _populated(node):
    """A message expression counts as populated when it is there at all; `hint` counts
    only when it is not an empty literal, because `hint=""` is how a site says "no hint"."""
    if node is None:
        return None
    if _const_str(node) == "":
        return None
    return ast.dump(node)


def _emission_sites():
    """code -> [_Site], for every construction that names a code as a literal.

    Two shapes carry a code at head: a CALL with a `code="E…"` keyword (`make_diagnostic`,
    the per-phase `_error`/`_make_error` builders, `TypeCheckError(...)`) and a DICT
    literal with a `"code"` key (the JSON diagnostics a tool hands a host). A function
    PARAMETER default (`def _error(..., code: str = "E1000")`) is neither, so it is not a
    site — which is the right answer: a default is the code a caller gets when it asks for
    none, not a place the product decided to emit one.
    """
    out = collections.defaultdict(list)
    for path, rel in _product_files():
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        try:
            tree = ast.parse(src)
        except SyntaxError:               # pragma: no cover - product code parses
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                kw = {k.arg: k.value for k in node.keywords if k.arg}
                code = _const_str(kw.get("code"))
                if code is None or not _CODE.fullmatch(code):
                    continue
                msg = kw.get("message")
                if msg is None and node.args:
                    msg = node.args[0]    # the builders take the message positionally
                out[code].append(_Site(rel, node.lineno, _callee_name(node),
                                       _populated(msg), _populated(kw.get("hint")),
                                       _const_str(kw.get("severity"))))
            elif isinstance(node, ast.Dict):
                d = {_const_str(k): v for k, v in zip(node.keys, node.values)
                     if _const_str(k) is not None}
                code = _const_str(d.get("code"))
                if code is None or not _CODE.fullmatch(code):
                    continue
                out[code].append(_Site(rel, node.lineno, "<dict>",
                                       _populated(d.get("message")),
                                       _populated(d.get("hint")),
                                       _const_str(d.get("severity"))))
    return out


# ── 1. The ratchet ────────────────────────────────────────────────────

# The codes that had no test when this pin was last moved — the debt, written down. The
# count is pinned beside it because the count is what the ratchet promises: it may go DOWN
# and never up. Raising either is not a fix; a code arriving without a test is the thing
# the row exists to catch, and the failure names it.
#
# Moving the pin is a two-line edit in ONE direction: drop the code that gained a test from
# the set, and lower the number to match. The row tells you both numbers when it reds.
_UNTESTED_PIN = 16
_UNTESTED_AT_PIN = frozenset("""
    E1000 E3000
    E3100 E3900
    E6000 E6001 E6002 E6004 E6005 E6006 E6030 E6040 E6050 E6051 E6060
    E9001
""".split())

# Why each of the sixteen is still here, so the next reader does not re-derive it:
#
# * `E1000` / `E3000` are the `code=` DEFAULT of `LexerError` / `TypeCheckError`. No call
#   site omits `code=`, so nothing in the product constructs either. A test could only
#   build the exception itself, which would test the test.
# * `E3100` ("unknown type name") cannot fire from source: a declared type must be a type
#   KEYWORD, and every keyword is a key of `TYPE_NAME_MAP`. The guard covers a caller that
#   builds the AST directly.
# * `E3900` ("an array literal outside a declaration") likewise: `{...}` is parsed only as
#   a declaration's initializer, so the expression handler has no path from source.
# * `E6xxx` are the interpreter's. They need an EXECUTION — a compiled program and real
#   tensors — not a `check` or a `compile`, and most of them are defensive branches the
#   type checker forecloses before the interpreter is reached.
# * `E9001` comes from a fused tool's preflight, which needs a host's tool manifest.
#
# Four of those are worth an issue rather than a test: a documented code the product
# cannot emit is a promise to a host that nothing keeps.


def test_simp6_untested_error_codes_only_go_down(r: SubTestResult):
    """Pin the codes no test names; a new one reds, and retiring one moves the pin."""
    if len(_UNTESTED_AT_PIN) != _UNTESTED_PIN:
        r.fail("the pin disagrees with itself",
               f"_UNTESTED_PIN is {_UNTESTED_PIN} but _UNTESTED_AT_PIN holds "
               f"{len(_UNTESTED_AT_PIN)} code(s). They are one fact written twice; fix both.")
        return

    product = _codes_named_by_product()
    tested = _codes_named_by_tests()
    untested = sorted(c for c in product if c not in tested)

    fresh = sorted(set(untested) - _UNTESTED_AT_PIN)
    if fresh:
        r.fail(
            "error-code coverage ratchet",
            f"{len(fresh)} code(s) have no test naming them and are not in the pinned "
            f"backlog: {' '.join(fresh)}. "
            f"(All {len(untested)} untested now: {' '.join(untested)}.) "
            f"Add a row that triggers the code through the public surface and asserts "
            f"it — tests/test_simp6_error_code_rows.py groups them the way the codes are "
            f"grouped. If the code genuinely cannot be triggered, widen the pin ONLY with "
            f"the reason written beside it.")
        return

    retired = sorted(_UNTESTED_AT_PIN - set(untested))
    if retired:
        r.fail(
            "error-code coverage ratchet is stale",
            f"{len(retired)} code(s) in the pinned backlog now have a test: "
            f"{' '.join(retired)}. Move the pin: drop them from _UNTESTED_AT_PIN and set "
            f"_UNTESTED_PIN to {len(untested)}. A pin that no longer describes the tree "
            f"stops being a ratchet.")
        return

    r.ok(f"{len(untested)} of {len(product)} error codes have no test naming them, exactly "
         f"the pinned backlog: {' '.join(untested)}")


# ── 2. Families ───────────────────────────────────────────────────────

# One code raised from several positions, each with its own message, is a FAMILY: the code
# names a CLASS of mistake and the message names the instance. That is a design, not a
# duplication — but it is indistinguishable from copy-paste unless somebody writes down
# which codes are meant to be families. These are, with the reason and the modules the
# sites live in; a family that stops being one (a site removed, or a site moved to another
# module) reds here rather than being noticed by nobody.
_FAMILIES = {
    "E2010": (
        "The parser's 'a statement ends with `;`' code. Every position where a statement "
        "can end is a separate raise, because the message names WHAT ended (an assignment, "
        "a declaration, an expression) and the caret points at that statement's last token.",
        ("tex_compiler/parser.py",),
    ),
    "E5003": (
        "A call whose ARGUMENTS are wrong — raised once per rule a call can break, and "
        "per function that has its own rule (`len` on a number, `cross` on a vec2, "
        "`select` on a vector condition, a user function's arity and per-argument types). "
        "One shared message could only say 'bad argument', which is the one thing the "
        "caller already knows.",
        ("tex_compiler/type_checker.py",),
    ),
    "E3402": (
        "Matrix arithmetic that does not typecheck. Each site names the shape rule that "
        "was broken (mismatched matrix sizes, a mat3 without a vec3/vec4, the operands in "
        "the wrong order) and suggests the fix for THAT rule.",
        ("tex_compiler/type_checker.py",),
    ),
    "E3200": (
        "'This value is not that type' — the type checker's assignability failure, raised "
        "wherever a value meets a declared type: a variable's initializer, a parameter's "
        "default, an assignment, an output binding inferred as string in one branch and "
        "numeric in another. Each site knows which two types it had and says so.",
        ("tex_compiler/type_checker.py",),
    ),
}


def test_simp6_declared_families_are_families(r: SubTestResult):
    """Each declared family really is several sites, in the declared modules, with
    several distinct messages — the three things that make it a family and not a copy."""
    sites = _emission_sites()
    for code in sorted(_FAMILIES):
        reason, modules = _FAMILIES[code]
        got = sites.get(code, [])
        if len(got) < 2:
            r.fail(f"{code} is declared a family",
                   f"but the tree has {len(got)} construction site(s): "
                   f"{[f'{s.file}:{s.line}' for s in got]}. A one-site code is not a family — "
                   f"remove the declaration, or restore the sites.")
            continue
        files = sorted({s.file for s in got})
        if files != sorted(modules):
            r.fail(f"{code}'s family declaration names the wrong modules",
                   f"declared {sorted(modules)}, tree has {files} "
                   f"({[f'{s.file}:{s.line}' for s in got]}). Update the declaration, and "
                   f"say in the reason why the code now crosses that module.")
            continue
        messages = {s.message for s in got}
        if len(messages) < 2:
            r.fail(f"{code} is declared a family",
                   f"but all {len(got)} sites build the SAME message expression: "
                   f"{[f'{s.file}:{s.line}' for s in got]}. Sites that say the same thing "
                   f"are a duplicate to collapse, not a family to declare.")
            continue
        assert reason                     # a family without a written reason is a guess
        r.ok(f"{code}: a family of {len(got)} sites in {files}, "
             f"{len(messages)} distinct message expressions")


# ── 3. Cross-file codes ───────────────────────────────────────────────

# Codes emitted from more than one MODULE with different message text. A family inside one
# module is read by whoever edits that module; a code that crosses modules is not, and the
# two halves can drift into two different editor experiences under one code. `E0000` is the
# clearest case: it is TEX's "internal error" contract and it is built by a public API, by
# the compiler's fallback translator and by a tool's JSON reply, three authors who never
# read each other. What every site must agree on is the part a consumer BRANCHES on.
_CROSS_FILE = ("E0000", "E2000", "E2002")


def _implied_severity(code):
    """A code's letter IS its severity: `E` errors, `W` advisories. A site may spell it
    out, and then it must spell out the same thing."""
    return "error" if code.startswith("E") else "warning"


def test_simp6_cross_file_codes_agree_on_severity_and_hint_shape(r: SubTestResult):
    """For each code emitted from more than one module: same severity at every site, and
    the same answer at every site to "does this diagnostic carry a hint?".

    The full field set is NOT compared. The sites go through different builders — one
    takes `loc`/`phase`, one is a raw dict for a JSON reply — so "same fields populated"
    can only honestly mean the fields a consumer reads: the severity it branches on, the
    message it shows, and whether there is a hint to draw under it.
    """
    sites = _emission_sites()
    for code in _CROSS_FILE:
        got = sites.get(code, [])
        where = [f"{s.file}:{s.line}" for s in got]
        files = sorted({s.file for s in got})
        if len(files) < 2:
            r.fail(f"{code} is listed as a cross-file code",
                   f"but the tree builds it in {files or '[]'} ({where}). Either a site "
                   f"moved and the list is stale, or the code stopped crossing modules — "
                   f"drop it from _CROSS_FILE with a line saying which.")
            continue

        want = _implied_severity(code)
        spelled = {s.severity for s in got if s.severity is not None}
        if spelled - {want}:
            r.fail(f"{code} sites disagree on severity",
                   f"{sorted(spelled)} across {where}; the code's letter says '{want}'. "
                   f"A consumer branches on severity, so two sites under one code must not "
                   f"disagree about whether it is an error.")
            continue

        hinted = {s.file + ":" + str(s.line) for s in got if s.hint is not None}
        if hinted and len(hinted) != len(got):
            r.fail(f"{code} sites disagree on hint shape",
                   f"{sorted(hinted)} carry a hint, {sorted(set(where) - hinted)} do not. "
                   f"One code that sometimes has a help line and sometimes does not is two "
                   f"different editor experiences under one contract.")
            continue

        missing = [f"{s.file}:{s.line}" for s in got if s.message is None]
        if missing:
            r.fail(f"{code} sites disagree on message",
                   f"{missing} build no message. Every diagnostic leads with its message.")
            continue

        r.ok(f"{code}: {len(got)} sites across {files} agree — severity '{want}', "
             f"hint {'at every site' if hinted else 'at no site'}")
