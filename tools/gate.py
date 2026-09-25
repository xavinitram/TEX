#!/usr/bin/env python3
"""One command, one verdict: the gate an implementer runs before handing work back.

WHY THIS FILE EXISTS
--------------------
The gate set used to be prose spread over three documents: a cheap list, a CI shape, a
whole-suite run through a six-line wrapper that every reader retyped, and a known-red
allowlist applied by eye. Two consequences, both measured over thirteen lanes:

  * the whole-suite run returned a non-zero exit code on **every one of them**, because one
    standing red was a test bug nobody owned — so its exit code carried no information and
    "green" meant "read the error list and agree it is the expected one"; and
  * the same pair of whole-suite runs was re-run after a rebase for information the second
    run superseded, about 15 % of all the suite time spent.

So: one entry point, two tiers, the allowlist held as DATA next to the tests it names, and a
verdict keyed on the tree so an unchanged tree is not re-measured. The exit code is the
verdict:

    0   GREEN
    1   RED       — at least one failure that the allowlist does not name
    2   GREEN, but the allowlist is STALE — it names a red that did not fire. A stale
                    allowlist is the failure mode this tool exists to remove, so it is
                    reported in the exit code rather than in a paragraph.

TIERS — and what each one actually proves
-----------------------------------------
`--tier cheap` runs the eight ratchets that answer in seconds: the no-numpy ban, the LOC and
headroom ratchets, the archive-surface ratchet, the host-path counts pins, TST-7's runner
drift check, the private-root lint over the tracked set, the shared-stash law, and the
local-only-path lint over the tracked set (LINT-1). Every one of them is a strict
SUBSET of the full tier; they are kept for feedback latency, not for coverage, and this tool
says so out loud. It also excludes `timing` (below) — belt and braces, since none of the
eight ratchet files carries that marker today, but a future one might.

**The `timing` marker** (v0422-gatehyg / TRK-168, TRK-14, TRK-75). A test asserting a
wall-clock ratio, a speedup or a deadline belongs to a sitting on a quiet, dedicated
reference box, not to a gate any lane's shared laptop runs unattended — the shared box is
exactly what made those three tracker rows flake. Every tier therefore adds `and not timing`
to its `-m` expression, and reports how many `timing` tests it deselected as its own number
(`<leg>-timing-deselected N` on the leg's line) rather than folding it into pytest's own
combined "N deselected" count, which mixes every active marker into one figure. Run them
deliberately with `-m timing` on the box they are meant for. A per-program TIMEOUT used as a
hang guard (e.g. `test_integration.py::test_example_files_compiled`) is a different thing —
it stays in the gate, unmarked, with a bound wide enough not to trip on a merely slow box.

`--tier full` runs cheap first (cheapest first, and it aborts there if cheap is red unless
`--keep-going`), then the two whole-suite legs that are NOT subsets of each other:

  * **ci-shape** — a second interpreter, ideally the Python version CI uses and one with no
    embedding host installed, run from the package ROOT so the host is off `sys.path`, with
    `CUDA_VISIBLE_DEVICES=-1`, `-m "not slow and not timing"`, `-p no:cacheprovider`. It is
    the only leg that can catch a test which assumes a host or a GPU. It runs on whatever OS
    you are on, so it cannot catch a line-ending or toolchain difference — say that when
    quoting it.

    Which interpreter, in order: `--ci-python`, then the `TEX_CI_PYTHON` environment variable,
    then the interpreter running this script. **No path is hard-coded here**: where a second
    Python lives is a property of a particular machine, and this file is published. The last
    resort still runs the leg — the shape alone is worth something — but it is neither a second
    version nor a host-free installation, so the leg's `proves:` line says which interpreter it
    used and, when it fell back, that it proves less. Set `TEX_CI_PYTHON` once per box and the
    question stops arising. This tool only ever RUNS that interpreter; it never installs into it.
  * **canonical** — the embedded interpreter, from the package's PARENT, through
    `tools/canonical_harness.py` (the v3 NodeOutput wrapper disarmed), `-X utf8`. It is the
    only leg that exercises CUDA and the host-present path.

Optionally `--counts-baseline PATH` adds the structural counts leg, run at the gate shape
(96^2 / 48^2 / 4 ticks, CPU) with `--counters-only`: its verdict counts API rows and reports
the frame census outside the exit code, because the frame rows move for every lawful change
that adds a call or moves a module. The baseline must be a `--save` taken at that same shape.

`--tier touched` (SPLIT-E) is a THIRD, standalone tier — a lane's own gate, run before handing
back, in between `cheap`'s eight-ratchet feedback latency and `full`'s whole-suite landing
cost. It selects, on the canonical harness, the full test files whose NAMES or IMPORTS relate
to what this branch touched against `--base` (default `origin/main`): every file `git diff
<base>..HEAD --name-only` names that is itself under `tests/` (its own name IS the relation —
a modified test always re-runs), every OTHER test file whose own imports resolve to a product
module the diff touched (`tools/gate.py`'s `_test_module_refs` against `_touched_module`, a
simple, documented, AST-read mapping — no import-graph transitive closure, no test-name
guessing), and, ALWAYS, six cheap ratchets a cheap-only lane has no standing reason to ever
run and has kept missing as a result: the docs (`test_v018_docs.py`), citation
(`test_simp5_citations.py`), mutation (`test_mut1_harness.py`), embedding-host seam
(`test_seam45_embedding_host_seam.py`), skip-budget (`test_simp3_skip_budget.py`) and LOC/
headroom-floor (`test_v017_phase2.py`) ratchets. It carries the same dead-leg guard as every
other leg (`_run`'s `expect_collect`, on by default): a selection that collects zero tests is
a RED, not a silent GREEN, and an unresolved `--base` (an unfetched ref, a typo) says so in
the leg's `proves:` line rather than quietly degrading to the ALWAYS set with no explanation.
It is not a landing gate and never substitutes for `full`.

CACHE
-----
A verdict is a claim about a tree **as read by a particular set of interpreters**, so all of
that is in the key: a sha256 over every tracked and untracked-not-ignored file's bytes, the
tier, and, for every interpreter that tier runs, its RESOLVED absolute path *and* its
`sys.version`. A re-run under the same conditions prints the cached verdict with the timestamp
of the run that produced it, and the interpreters it belonged to, and exits with the same code;
`--no-cache` forces a real run and refreshes the entry. The cache lives OUTSIDE the repository
on purpose — a cache file inside it would change the very hash it is keyed on.

The interpreter half is not decoration. The key used to carry `os.path.basename(ci_python)`,
which on Windows is `python.exe` for every interpreter there has ever been, so a verdict
measured with one was served for another — it handed back a RED that belonged to a different
Python, and only `--no-cache` got past it. The version is in there too, for the path whose
interpreter was upgraded underneath it.

Note which interpreters a tier actually uses. `--tier cheap` runs only `--python`, so that is
all its key carries. `--tier full` also runs the CI-shape interpreter — and with `TEX_CI_PYTHON`
unset and no `--ci-python`, **that leg runs under `sys.executable`**: the same interpreter as
the other legs, proving less, and the key and the printed line say exactly that by naming the
same path twice.

`TEX_CACHE_DIR` (a leg's TEX-side cache) is scratch and wiped fresh every run, on purpose — but
`TORCHINDUCTOR_CACHE_DIR` (its torch.compile/Inductor kernel sub-cache) is pinned to a PERSISTENT
per-leg directory instead (`_inductor_cache_dir`, next to the verdict cache above), reused across
runs: fewer never-before-seen compiled DLLs per run is fewer chances for an OS reputation check
(Windows Application/Smart App Control) to block one (V045-FIX; `GATE-SAC.md`).

`tools/` is excluded from the published archive (`.comfyignore`), so nothing here ships.
"""
from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                        # .../TEX_Wrangle
_PARENT = os.path.dirname(_PKG)
_HARNESS = os.path.join(_HERE, "canonical_harness.py")
_ALLOWLIST = os.path.join(_PKG, "tests", "known_reds.json")


def _importable_as_tex_wrangle() -> bool:
    """`tests/helpers.py` (every test file's own import path) does `from TEX_Wrangle.<mod>
    import ...` literally, against a directory named exactly `TEX_Wrangle` under the
    package's parent -- whether that is the package directory itself (this checkout's own
    folder is named `TEX_Wrangle`) or a sibling that resolves to it (the ComfyUI layout: a
    `TEX_Wrangle` junction next to a `TEX` checkout). A checkout with neither -- a plain
    `git worktree add` into a scratch directory with no such sibling -- fails EVERY test at
    collection with a bare `ModuleNotFoundError`, before any leg's process can produce a
    diagnosable result. Checked once, up front, so that shape gets a clear refusal instead of
    a wall of tracebacks with no verdict line."""
    return os.path.isdir(os.path.join(_PARENT, "TEX_Wrangle"))

#: The ratchets that answer in seconds. Each is a strict subset of both whole-suite legs.
_CHEAP = [
    ("no-numpy ban", "tests/test_no_numpy_ban.py"),
    ("LOC + headroom ratchets", "tests/test_v017_phase2.py"),
    ("archive surface ratchet", "tests/test_pub1_archive.py"),
    ("host-path counts pins", "tests/test_bench2_counts.py"),
    ("TST-7 runner drift", "tests/test_v017_phase1.py"),
    # Here because of what it guards: this file once shipped an absolute path into one
    # machine's project root and every gate below passed, because the shipped-surface
    # ratchets skip `tools/` on purpose. The lint scans the TRACKED set instead, and it
    # belongs in the tier a change is read against rather than three minutes downstream.
    ("private-root lint", "tests/test_simp3_no_machine_paths.py"),
    # v0422-gatehyg / TRK-128: a stash in ANY worktree of this repository is live for
    # every one of them (one shared `.git`), and the fast tier is the gate every lane
    # actually runs before it acts -- the cheapest place to catch it before it matters.
    ("shared-stash law", "tests/test_v0422_no_shared_stash.py"),
    # LINT-1 (v0.43.0 rider (c)): SIMP-3's sibling gap -- a path that is per-*repository-
    # checkout* (this project's own local, unpushed working area) rather than per-person.
    # Pure text scan, no compile, same cost class as the private-root lint beside it.
    ("local-only-path lint", "tests/test_lint1_no_local_only_path_refs.py"),
]

#: SPLIT-E's `--tier touched`: the fixed ratchets a lane's own selection ALWAYS carries,
#: because they are cheap and are exactly what a cheap-only lane kept missing (a lane briefed
#: for `--tier cheap` plus "the files your change touches" has no standing reason to ever run
#: the docs/citation/mutation/seam/skip-budget ratchets, so drift in any of them shipped
#: undetected until the orchestrator's own `--tier full`). Each is a whole-file ratchet, not a
#: per-module test, so "does this touch it" is not a question of imports -- it always applies.
_ALWAYS_TOUCHED = [
    ("docs map-drift + citation-adjacent doc checks (DOC-7)", "tests/test_v018_docs.py"),
    ("citation budget (SIMP-5)", "tests/test_simp5_citations.py"),
    ("mutation harness (MUT-1)", "tests/test_mut1_harness.py"),
    ("embedding-host seam (SEAM-45)", "tests/test_seam45_embedding_host_seam.py"),
    ("skip-budget ratchet (SIMP-3)", "tests/test_simp3_skip_budget.py"),
    ("LOC + headroom floors (REG-2/ENG-14)", "tests/test_v017_phase2.py"),
]

#: Product source directories a touched path can resolve a dotted module name under (root-level
#: `tex_*.py` needs no entry -- see `_touched_module`). `tests/`, `tools/`, `benchmarks/`,
#: `docs/`, `.github/`, `examples/`, `assets/`, `editor_build/` and dotfiles/markdown are
#: deliberately NOT product modules a test would import as `TEX_Wrangle.<dotted>`; a change
#: confined to those is caught by the ALWAYS set above, never by the import-matching below.
_PRODUCT_SUBDIRS = ("tex_compiler", "tex_runtime", "tex_io")


def _touched_module(path: str) -> str | None:
    """A touched repo-relative path -> the dotted module name a test file would import it as
    (never carrying the `TEX_Wrangle.` package prefix — e.g. `tex_engine`,
    `tex_runtime.compiled`), or `None` when the path is not an importable product module."""
    p = path.replace("\\", "/")
    if not p.endswith(".py") or p in ("__init__.py",):
        return None
    if "/" not in p:
        return p[:-3]                      # root-level tex_*.py -> tex_*
    top, rest = p.split("/", 1)
    if top not in _PRODUCT_SUBDIRS:
        return None                        # tests/, tools/, benchmarks/, docs/, ... -- not a module
    return top + "." + rest[:-3].replace("/", ".")


def _test_module_refs(path: str) -> set:
    """Every `TEX_Wrangle.<dotted>` module a test FILE's own imports could resolve to, as
    dotted paths with the `TEX_Wrangle.` prefix stripped (matching `_touched_module`'s
    spelling) -- covers the two import shapes every file in `tests/` actually uses:
    `from TEX_Wrangle import a[, b...]` (each name is a top-level submodule) and
    `from TEX_Wrangle.a.b import c` / `import TEX_Wrangle.a.b` (the MODULE is `a.b`; `c` is
    one of its attributes, not resolved any further -- a test importing a name out of a
    module that touched still needs to re-run, so under-resolving here is the safe direction).
    Returns the empty set on anything that fails to parse, never raises -- a selection helper
    that could crash the gate on a stray test file is worse than one that just skips it."""
    try:
        tree = ast.parse(open(path, encoding="utf-8").read())
    except Exception:
        return set()
    refs = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "TEX_Wrangle":
                refs.update(alias.name for alias in node.names)
            elif node.module.startswith("TEX_Wrangle."):
                refs.add(node.module[len("TEX_Wrangle."):])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("TEX_Wrangle."):
                    refs.add(alias.name[len("TEX_Wrangle."):])
    return refs


def touched_files(base: str) -> list:
    """`git diff <base>..HEAD --name-only`, repo-relative paths, empty list on any git error
    (a bad or unfetched `base` ref) rather than a raised exception -- `select_touched_tests`
    degrades to the ALWAYS set alone in that case, and `run_touched`'s `proves:` line says so
    (never a silent, unexplained empty selection — the exact lie this file's docstring exists
    to remove elsewhere)."""
    out = _git("diff", f"{base}..HEAD", "--name-only")
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def select_touched_tests(base: str) -> tuple:
    """The `--tier touched` selection. Returns `(files, touched_mods, base_resolved)`:
    `files` is the sorted, de-duplicated list of `tests/*.py` relative paths to run;
    `touched_mods` is the set of dotted module names the diff touched (for the report line);
    `base_resolved` is False when `base` did not resolve (git error / unfetched ref), in which
    case `files` is still the ALWAYS set (never empty) but the caller should say so out loud.

    The rule, documented once here rather than re-derived per lane: a test file is selected
    when (a) it is itself one of the files the diff touched (its own name IS the relation —
    a modified test always re-runs itself), or (b) it imports a module the diff touched
    (`_test_module_refs` ∩ the touched dotted-module set), or (c) it is in the fixed
    `_ALWAYS_TOUCHED` list, unconditionally."""
    base_resolved = _git("rev-parse", "--verify", "--quiet", base).strip() != ""
    changed = touched_files(base) if base_resolved else []
    selected = {p for _, p in _ALWAYS_TOUCHED}
    touched_mods = set()
    for path in changed:
        p = path.replace("\\", "/")
        if p.startswith("tests/") and p.endswith(".py"):
            selected.add(p)
            continue
        mod = _touched_module(p)
        if mod:
            touched_mods.add(mod)
    if touched_mods:
        tests_dir = os.path.join(_PKG, "tests")
        for fn in sorted(os.listdir(tests_dir)):
            if not (fn.startswith("test_") and fn.endswith(".py")):
                continue
            rel = f"tests/{fn}"
            if rel in selected:
                continue
            if _test_module_refs(os.path.join(tests_dir, fn)) & touched_mods:
                selected.add(rel)
    return sorted(selected), touched_mods, base_resolved


#: Where the CI-shape interpreter is named, so this file names no machine's private layout.
_CI_PYTHON_ENV = "TEX_CI_PYTHON"

#: The second alternative used to be `(?:\d+ \w+,? ?)+ in [\d.]+s`: `\d+` and `\w+` both
#: accept digits, and the trailing `,? ?` was optional on both sides, so a tail that never
#: reaches " in <secs>s" (e.g. many "000 " repeats with no letters) let the engine re-split
#: the same run of digits between `\d+` and `\w+` in quadratically many ways before giving
#: up — a CodeQL-flagged inefficient regex. Pytest's own count words are always alphabetic
#: (`passed`, `failed`, `errors`, `warnings`, `deselected`, `skipped`, `xfailed`, `xpassed`),
#: so `[a-z]+` (case-insensitive) already covers every real word and shares no characters
#: with `\d+` — the ambiguity, not just this one exploit string, is gone. The repeated
#: clauses are joined by a literal ", " (what every pytest summary actually uses), never an
#: optional separator, so there is only one way to parse a match.
_SUMMARY_RE = re.compile(
    r"^[=\s]*\d+ (?:passed|failed|error|deselected|skipped)|"
    r"^\s*\d+ [a-z]+(?:, \d+ [a-z]+)* in [\d.]+s", re.I)


# ──────────────────────────────────────────────────────────────────────────────
# Tree identity
# ──────────────────────────────────────────────────────────────────────────────

def _git(*args, cwd=_PKG) -> str:
    try:
        return subprocess.run(["git", "-C", cwd, *args], capture_output=True,
                              text=True, timeout=120).stdout
    except Exception:
        return ""


def tree_hash() -> str:
    """sha256 over the bytes of every file git would show you, path included.

    Tracked files AND untracked-not-ignored ones: a lane that adds a test file has not
    committed it yet, and a cache that could not see it would hand that lane a stale GREEN.
    Ignored paths (the orchestration material in `.git/info/exclude`) are deliberately
    invisible, so writing a hand-back does not invalidate a verdict."""
    out = _git("ls-files", "-z", "--cached", "--others", "--exclude-standard")
    paths = sorted(p for p in out.split("\0") if p)
    h = hashlib.sha256()
    for rel in paths:
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        try:
            with open(os.path.join(_PKG, rel), "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
        except OSError:
            h.update(b"<unreadable>")
        h.update(b"\0")
    return h.hexdigest()


_VERSION_PROBE = "import sys;print(sys.version)"
_IDENTITY_MEMO: dict = {}


def interpreter_identity(path: str) -> tuple:
    """`(resolved absolute path, sys.version)` for an interpreter, queried once per process.

    Both halves are needed and neither is enough. The PATH distinguishes two interpreters that
    happen to be named the same thing — on Windows every one of them is called `python.exe`,
    which is how a verdict measured with one came to be served for another. The VERSION
    distinguishes one path whose interpreter was upgraded underneath it, which no amount of
    path comparison can see.

    A path that cannot be run still yields an identity: its resolved path plus a note saying
    the version is unknown. That is deliberate — an interpreter this tool cannot query is one
    whose verdict should not be shared with any other, and returning something unique keeps the
    cache honest rather than collapsing every unqueryable path onto one key."""
    memo_key = os.path.normcase(os.path.abspath(path))
    hit = _IDENTITY_MEMO.get(memo_key)
    if hit is not None:
        return hit
    real = os.path.normcase(os.path.realpath(path))
    try:
        proc = subprocess.run([path, "-c", _VERSION_PROBE], capture_output=True,
                              text=True, timeout=120)
        version = (" ".join(proc.stdout.split()) if proc.returncode == 0 and proc.stdout.strip()
                   else f"<unqueryable: rc {proc.returncode}>")
    except Exception as e:
        version = f"<unqueryable: {type(e).__name__}>"
    hit = _IDENTITY_MEMO[memo_key] = (real, version)
    return hit


def cache_key(tree: str, tier: str, interpreters, with_counts: bool) -> str:
    """The key a verdict is stored under: the tree, the tier, and WHO measured it.

    `interpreters` is `[(role, path), …]` — every interpreter this tier will run. A verdict is
    a claim about a tree *as read by a particular set of interpreters*, so all of them belong
    in the key; the basename alone does not distinguish them (see `interpreter_identity`).
    Hashed, so the cache file's keys stay one line whatever the paths look like — the readable
    identities travel in the record and are printed with the cached verdict."""
    parts = [f"tree={tree}", f"tier={tier}", f"counts={int(bool(with_counts))}"]
    for role, path in interpreters:
        real, version = interpreter_identity(path)
        parts.append(f"{role}={real}|{version}")
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def describe_interpreters(interpreters) -> str:
    """The readable half of `cache_key` — what a reader needs to trust a cached verdict."""
    out = []
    for role, path in interpreters:
        real, version = interpreter_identity(path)
        out.append(f"{role} {real} ({version.split(' ')[0]})")
    return " | ".join(out)


def head_label() -> str:
    sha = _git("rev-parse", "--short", "HEAD").strip() or "?"
    return sha + ("-dirty" if _git("status", "--porcelain").strip() else "")


def _cache_path() -> str:
    env = os.environ.get("TEX_GATE_CACHE")
    if env:
        return env
    return os.path.join(tempfile.gettempdir(), "tex-gate-verdicts.json")


def _inductor_cache_dir(leg_name: str) -> str:
    """PERSISTENT torch.compile/Inductor kernel cache for `leg_name` — deliberately NOT under
    `scratch` (V045-FIX). `TEX_CACHE_DIR` (below) is wiped fresh every run on purpose, so it
    doubled as a fresh `TORCHINDUCTOR_CACHE_DIR` too: every leg that forces a CPU/CUDA compile
    (the tiered-noise promotion tests) therefore built and loaded a never-before-seen native
    kernel on EVERY gate run, which is exactly the shape a Windows Application/Smart App
    Control policy's reputation check can intermittently block (`GATE-SAC.md`; three
    occurrences the same day). Reusing compiled kernels across runs means far fewer
    never-seen DLLs for that check to see, so this lives next to the verdict cache above:
    same convention (`TEX_GATE_CACHE`-style env override, else a fixed name under the OS temp
    dir — portable, and outside both the repository and the per-run scratch dir), one
    directory per LEG so no two legs' compiled artifacts collide. Never wiped by this file, by
    `--no-cache` or by a `--scratch` change — only `TEX_CACHE_DIR`'s per-run freshness is about
    the verdict; torch's own cache is content-addressed by the generated source, so a changed
    kernel gets a new entry rather than serving a stale one. A test that must observe an
    actually-cold Inductor compile gets its OWN fresh directory locally instead of relying on
    this one being empty — see `tests/helpers.py` and its callers."""
    root = os.environ.get("TEX_GATE_INDUCTOR_CACHE") or \
        os.path.join(tempfile.gettempdir(), "tex-gate-inductor-cache")
    d = os.path.join(root, leg_name)
    os.makedirs(d, exist_ok=True)
    return d


def _cache_read(key: str) -> dict | None:
    try:
        with open(_cache_path(), "r", encoding="utf-8") as fh:
            return json.load(fh).get(key)
    except Exception:
        return None


def _cache_write(key: str, record: dict) -> None:
    path = _cache_path()
    try:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            data = {}
        data[key] = record
        # Keep the file from growing without bound; verdicts are cheap to recompute.
        if len(data) > 64:
            for k in sorted(data, key=lambda k: data[k].get("at", ""))[:len(data) - 64]:
                data.pop(k, None)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1)
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# The allowlist, as data
# ──────────────────────────────────────────────────────────────────────────────

def load_allowlist() -> list:
    """`tests/known_reds.json` -> the entries, validated. A missing file is an empty list."""
    try:
        with open(_ALLOWLIST, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except FileNotFoundError:
        return []
    entries = doc.get("entries", [])
    for i, e in enumerate(entries):
        missing = [k for k in ("id", "reason", "when", "condition", "owner") if k not in e]
        if missing:
            raise SystemExit(f"{_ALLOWLIST}: entry {i} is missing {missing}; every entry "
                             f"states the test id, why it is red, the machine-readable "
                             f"condition it is expected under, the human sentence for that "
                             f"condition, and who removes it")
    return entries


def _applies(entry: dict, ctx: dict) -> bool:
    """AND over a closed vocabulary — an entry that cannot be evaluated does not apply."""
    for tok in entry.get("when", []):
        if tok == "always":
            continue
        elif tok == "cuda" and not ctx["cuda"]:
            return False
        elif tok == "no_cuda" and ctx["cuda"]:
            return False
        elif tok.startswith("leg:") and tok[4:] != ctx["leg"]:
            return False
        elif tok not in ("always", "cuda", "no_cuda") and not tok.startswith("leg:"):
            return False
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Running a leg
# ──────────────────────────────────────────────────────────────────────────────

class Leg:
    def __init__(self, name: str, proves: str):
        self.name, self.proves = name, proves
        self.rc = None
        self.summary = "not run"
        self.failures: list = []
        self.collected: set = set()
        #: `{nodeid: assertion text}` for every id in `failures`, from the junit report's own
        #: `<failure>`/`<error>` element -- so a red is diagnosable from this leg's own printed
        #: log without a rerun. Two whole-suite reds went undiagnosable before this: the log
        #: named only the id, and the process that could explain it had already exited.
        self.failure_text: dict = {}
        self.seconds = 0.0
        #: How many `@pytest.mark.timing` tests this leg's own target excluded via
        #: `not timing`, or None when a leg carries no marker filter at all / the count
        #: itself could not be taken. Never silently folded into pytest's own combined
        #: "N deselected" (which mixes every active marker into one number) -- reported
        #: as its own figure so a timing test can never disappear unnoticed (v0422-gatehyg).
        self.timing_deselected: int | None = None


def _mark_infra_red(leg: "Leg", detail: str) -> None:
    """Turn a leg with nothing to red on into a real, un-allowlistable red.

    `judge()` reads only `leg.failures`, never `leg.rc` -- so a leg whose process died before
    it ever produced a parseable result (a nonzero rc with no junit report at all, e.g. a
    `conftest.py` import error that exits pytest before `pytest.main()` runs; or an rc-0 leg
    that collected nothing, e.g. every requested file got deselected, or the junit it wrote
    has zero `<testcase>` rows) used to fall through to `judge()` as a clean pass -- GREEN
    with a nonzero exit code sitting right next to it. Shared by `_run` and `run_counts`, the
    two places that discovered this independently (`run_counts`'s `<counts:infra-rc…>` id was
    first): give the leg a synthetic id no allowlist entry can name, so it always lands in
    `real` instead. Never fires on a leg that already parsed a real failure -- callers only
    reach this when `leg.failures` is empty."""
    infra_id = f"<{leg.name}:infra-rc{leg.rc}>"
    leg.failures = [infra_id]
    leg.failure_text[infra_id] = detail[:300]


def _stdout_detail(stdout: str) -> str:
    """The last non-blank line of a process's stdout, or a fixed note when there is none --
    the one-line explanation `_mark_infra_red` attaches to its synthetic id."""
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    return lines[-1] if lines else "(no output)"


def _parse_junit(path: str) -> tuple:
    """`(failing ids, every collected id, {failing id: assertion text})`, as
    `tests/<file>.py::<test>`.

    Exact, from the report pytest writes, not scraped from stdout. The COLLECTED set matters
    as much as the failing one: an allowlist entry is only stale if the leg actually ran the
    test it names, otherwise the cheap tier — which collects five files — would call every
    entry for a sixth file stale.

    The TEXT half exists so a red is diagnosable from THIS report alone: two whole-suite reds
    on the canonical leg went undiagnosable because the log recorded only the node id and the
    process had already exited by the time anyone looked. `message` is pytest's one-line
    summary of the assertion; when it is empty (some `error` nodes carry the text only in the
    body) the last non-blank line of the body is used instead — usually the assertion itself,
    never the whole traceback."""
    failing, seen, text = [], set(), {}
    try:
        root = ET.parse(path).getroot()
    except Exception:
        return failing, seen, text
    for case in root.iter("testcase"):
        nodeid = _nodeid_of(case)
        seen.add(nodeid)
        node = next((case.find(t) for t in ("failure", "error")
                    if case.find(t) is not None), None)
        if node is not None:
            failing.append(nodeid)
            msg = (node.get("message") or "").strip()
            body = (node.text or "").strip()
            snippet = msg or next((ln.strip() for ln in reversed(body.splitlines())
                                   if ln.strip()), "")
            text[nodeid] = snippet[:300]
    return sorted(set(failing)), seen, text


def _nodeid_of(case) -> str:
    """A junit `<testcase>` -> the node id the allowlist is written in.

    The `file` attribute is optional and this box's pytest does not emit it, so the path is
    recovered from the dotted `classname` by asking the DISK which prefix of it is a file —
    which also splits a class-based id correctly without guessing at capitalisation."""
    name = str(case.get("name"))
    f = (case.get("file") or "").replace("\\", "/")
    if f:
        return f"{'tests/' + f.split('/tests/', 1)[1] if '/tests/' in f else f}::{name}"
    parts = [p for p in (case.get("classname") or "").split(".") if p]
    for i in range(len(parts), 0, -1):
        rel = "/".join(parts[:i]) + ".py"
        if os.path.isfile(os.path.join(_PKG, rel)):
            return "::".join([rel, *parts[i:], name])
    return "::".join([*parts, name]) if parts else name


def _summary_of(stdout: str) -> str:
    for line in reversed([ln.strip() for ln in stdout.splitlines() if ln.strip()]):
        bare = line.strip("= ").strip()
        if _SUMMARY_RE.match(bare) or (" in " in bare and bare[0].isdigit()):
            return bare
    return "(no pytest summary line)"


def _run(leg: Leg, argv: list, cwd: str, env_extra: dict, scratch: str, verbose: bool,
         expect_collect: bool = True) -> Leg:
    """Run one whole-suite/ratchet leg and fill in `leg` from its junit report.

    `expect_collect` marks a leg where collecting zero tests is itself a failure rather than
    a legitimate empty run -- true for every leg this file defines today (cheap, ci-shape,
    canonical all expect to collect real tests every time), kept as a parameter rather than a
    constant so a future leg that is allowed to collect nothing can opt out explicitly."""
    cache = os.path.join(scratch, f"cache-{leg.name}")
    shutil.rmtree(cache, ignore_errors=True)
    os.makedirs(cache, exist_ok=True)
    junit = os.path.join(scratch, f"junit-{leg.name}.xml")
    # V045-FIX: TEX_CACHE_DIR stays fresh every run (the verdict's cold-cache promise is
    # unchanged) -- only TORCHINDUCTOR_CACHE_DIR is pinned to the PERSISTENT per-leg dir, so a
    # tiered-noise test that forces a compile reuses last run's kernel instead of building and
    # loading a never-before-seen one. `env_extra` can still override either, so an explicit
    # per-call value always wins.
    env = dict(os.environ, TEX_CACHE_DIR=cache, TORCHINDUCTOR_CACHE_DIR=_inductor_cache_dir(leg.name))
    env.update(env_extra)
    t0 = time.time()
    proc = subprocess.run(argv + [f"--junit-xml={junit}"], cwd=cwd, env=env,
                          capture_output=True, text=True, errors="replace")
    leg.seconds = time.time() - t0
    leg.rc = proc.returncode
    leg.summary = _summary_of(proc.stdout)
    leg.failures, leg.collected, leg.failure_text = _parse_junit(junit)
    if leg.rc and not leg.failures:
        # The process died with nothing to red on -- e.g. a `conftest.py` import error, which
        # exits pytest with a nonzero rc before `pytest.main()` ever runs and writes NO junit
        # report at all, so `_parse_junit` sees a missing file and returns nothing. Without
        # this, `judge()` (which reads only `leg.failures`) would print GREEN over a nonzero rc.
        _mark_infra_red(leg, _stdout_detail(proc.stdout))
    elif expect_collect and leg.rc == 0 and not leg.collected:
        # rc 0 but the leg proved nothing: no junit file (a wrapper that swallowed a nonzero
        # inner rc), or a junit with zero `<testcase>` rows (every target file deselected or
        # skipped at collection). A silent, empty "pass" is not a clean one either.
        _mark_infra_red(leg, _stdout_detail(proc.stdout))
    if verbose:
        print(f"\n--- {leg.name}: {' '.join(argv)} (cwd={cwd}) ---")
        print(proc.stdout[-8000:])
        if proc.stderr.strip():
            print(proc.stderr[-2000:])
    return leg


def _count_timing(run_argv: list, cwd: str, env_extra: dict) -> int | None:
    """How many tests `-m timing` alone selects, for the exact interpreter/harness/target
    a leg just ran with (its own argv, minus the tail after the target, with a collect-only
    `-m timing` tail substituted in). Reported so a `not timing` deselection is a NUMBER on
    the gate's own line, not folded into pytest's combined "N deselected" (which mixes every
    active marker into one figure the moment more than one applies). Collect-only, so this
    never executes a timing test itself. Returns None on any failure to collect -- never 0
    by accident, which would misreport as "no timing tests exist" when the census broke."""
    argv = run_argv + ["-q", "--collect-only", "-m", "timing", "-p", "no:cacheprovider"]
    try:
        proc = subprocess.run(argv, cwd=cwd, env=dict(os.environ, **env_extra),
                              capture_output=True, text=True, timeout=120)
    except Exception:
        return None
    if proc.returncode not in (0, 5):     # 5 = pytest's own "no tests collected"
        return None
    return sum(1 for ln in proc.stdout.splitlines() if "::" in ln)


def run_cheap(python: str, scratch: str, verbose: bool) -> Leg:
    leg = Leg("cheap", "the eight ratchets only — no whole-suite collection, "
                       "no host-absent lane, CUDA present")
    files = [f"TEX_Wrangle/{p}" for _, p in _CHEAP]
    base = [python, "-X", "utf8", _HARNESS, *files]
    argv = [*base, "-q", "-m", "not timing", "-p", "no:cacheprovider"]
    _run(leg, argv, _PARENT, {}, scratch, verbose)
    leg.timing_deselected = _count_timing(base, _PARENT, {})
    return leg


def run_touched(python: str, base_ref: str, scratch: str, verbose: bool) -> Leg:
    """`--tier touched` (SPLIT-E): a lane's own gate, run on the canonical harness like
    `cheap`/`canonical` above — the full test files whose names or imports relate to what
    this branch touched (`select_touched_tests`), plus the ALWAYS ratchets. Not a landing
    gate and not a substitute for `--tier cheap`/`full`; it exists for the coverage a
    cheap-only lane structurally cannot have (a docs/citation/mutation/seam/skip-budget
    drift, or a full-file regression in a module the lane's own change touched, that no
    `_CHEAP` ratchet and no "run the files you touched" instruction ever caught on its own)."""
    files, touched_mods, base_resolved = select_touched_tests(base_ref)
    proves = (f"base={base_ref}; {len(files)} test file(s) selected "
             f"({len(touched_mods)} touched module(s): {', '.join(sorted(touched_mods)) or '(none)'})")
    if not base_resolved:
        proves += (f" — WARNING: base ref {base_ref!r} did not resolve (unfetched? typo?); "
                   f"selection fell back to the ALWAYS-only set, which under-selects")
    leg = Leg("touched", proves)
    target = [f"TEX_Wrangle/{p}" for p in files]
    base = [python, "-X", "utf8", _HARNESS, *target]
    argv = [*base, "-q", "-m", "not timing", "-p", "no:cacheprovider"]
    _run(leg, argv, _PARENT, {}, scratch, verbose)
    leg.timing_deselected = _count_timing(base, _PARENT, {})
    return leg


def resolve_ci_python(explicit: str | None) -> tuple:
    """`(interpreter, where it came from)` for the CI-shape leg.

    In order: `--ci-python`, then `$TEX_CI_PYTHON`, then the interpreter running this script.
    No default path is written down here on purpose — a hard-coded one would name a particular
    machine's private layout, and this file is published. The last resort still RUNS the leg,
    because the shape is worth something even from one interpreter (it puts the package root on
    `sys.path` instead of its parent and hides the GPU), but it proves less: it cannot show that
    the suite passes on the Python version CI uses or without the embedding host installed. So
    it says which interpreter it used, every time, in the leg's `proves:` line."""
    if explicit:
        return explicit, "--ci-python"
    from_env = os.environ.get(_CI_PYTHON_ENV)
    if from_env:
        return from_env, f"${_CI_PYTHON_ENV}"
    return sys.executable, "fallback"


def run_ci_shape(ci_python: str, scratch: str, verbose: bool, source: str = "--ci-python") -> Leg:
    leg = Leg("ci-shape", "CPU-only, the embedding host off sys.path, the CI interpreter — "
                          "the only leg that catches a host or CUDA assumption; runs on this "
                          "OS, so it cannot see a line-ending or toolchain difference")
    leg.proves += f" [interpreter: {ci_python} (from {source})]"
    if source == "fallback":
        leg.proves += (f" — NOTE: no --ci-python and no ${_CI_PYTHON_ENV}, so the CI shape is "
                       f"being run by the CURRENT interpreter; it is not a second Python "
                       f"version and not a host-free installation, so it proves less")
    if not os.path.isfile(ci_python):
        leg.rc, leg.summary = 127, f"interpreter not found: {ci_python}"
        leg.failures = ["<ci-shape interpreter missing>"]
        return leg
    base = [ci_python, "-m", "pytest", "tests/"]
    argv = [*base, "-q", "-m", "not slow and not timing", "-p", "no:cacheprovider"]
    env_extra = {"CUDA_VISIBLE_DEVICES": "-1"}
    _run(leg, argv, _PKG, env_extra, scratch, verbose)
    leg.timing_deselected = _count_timing(base, _PKG, env_extra)
    return leg


def run_canonical(python: str, scratch: str, verbose: bool) -> Leg:
    leg = Leg("canonical", "the embedded interpreter with CUDA and the host present, the v3 "
                           "NodeOutput wrapper disarmed — the only leg that runs the GPU rows")
    base = [python, "-X", "utf8", _HARNESS, "TEX_Wrangle/tests"]
    argv = [*base, "-q", "-m", "not slow and not timing", "-p", "no:cacheprovider"]
    _run(leg, argv, _PARENT, {}, scratch, verbose)
    leg.timing_deselected = _count_timing(base, _PARENT, {})
    return leg


#: The shape the counts harness is a GATE in — the same one `tests/test_bench2_counts.py`
#: pins and `docs/host-path-counts.md` §5 calls the gate shape. It runs in seconds, where the
#: reporting shape (1024^2, 8 ticks, both devices) runs in minutes, and a gate nobody can
#: afford to run is not a gate. `--counts-baseline` must therefore be a `--save` taken at
#: exactly this shape; a baseline taken at another one reports every row as new.
_COUNTS_SHAPE = ["--device", "cpu", "--res", "96", "--window", "48", "--ticks", "4",
                 "--prof1", "off"]


def run_counts(python: str, baseline: str, scratch: str, verbose: bool) -> Leg:
    leg = Leg("counts", "API per-tick counts at the gate shape (96^2/48^2/4 ticks, CPU) "
                        "against a saved baseline; the frame census is reported outside the "
                        "verdict, and no CUDA row is measured at this shape")
    cache = os.path.join(scratch, "cache-counts")
    shutil.rmtree(cache, ignore_errors=True)
    os.makedirs(cache, exist_ok=True)
    argv = [python, "-X", "utf8", "TEX_Wrangle/benchmarks/host_path_counts.py",
            *_COUNTS_SHAPE, "--counters-only", "--compare", baseline]
    t0 = time.time()
    proc = subprocess.run(
        argv, cwd=_PARENT,
        env=dict(os.environ, TEX_CACHE_DIR=cache,
                 TORCHINDUCTOR_CACHE_DIR=_inductor_cache_dir(leg.name)),
        capture_output=True, text=True, errors="replace")
    leg.seconds, leg.rc = time.time() - t0, proc.returncode
    tail = [ln.strip() for ln in proc.stdout.splitlines() if "counter row(s) moved" in ln]
    leg.summary = tail[-1] if tail else "(no compare summary line)"
    if leg.rc:
        leg.failures = [ln.strip() for ln in proc.stdout.splitlines()
                        if ln.strip().startswith(("CHANGED", "NEW ROW", "GONE"))]
        if not leg.failures:
            # A nonzero rc with no parsed row diff is not a clean pass -- e.g. `compare()`'s
            # own shape-mismatch REFUSED (TRK-100), which prints neither CHANGED/NEW ROW/GONE.
            # See `_mark_infra_red` (shared with `_run`, which has the same class of bug on
            # the pytest legs).
            _mark_infra_red(leg, tail[-1] if tail else _stdout_detail(proc.stdout))
    if verbose:
        print(f"\n--- counts: {' '.join(argv)} ---\n{proc.stdout[-8000:]}")
    return leg


# ──────────────────────────────────────────────────────────────────────────────
# Verdict
# ──────────────────────────────────────────────────────────────────────────────

def judge(legs: list, allowlist: list, cuda: bool) -> dict:
    """Apply the allowlist EXPLICITLY: what it excused, and what it claimed and did not get."""
    real, excused, fired = [], [], set()
    for leg in legs:
        for nodeid in leg.failures:
            hit = next((e for e in allowlist
                        if e["id"] == nodeid and _applies(e, {"cuda": cuda, "leg": leg.name})),
                       None)
            if hit is None:
                real.append(f"{leg.name}:{nodeid}")
            else:
                excused.append(f"{leg.name}:{nodeid}  ({hit['reason']})")
                fired.add(id(hit))
    stale = [e for e in allowlist
             if id(e) not in fired
             and any(_applies(e, {"cuda": cuda, "leg": leg.name}) and e["id"] in leg.collected
                     for leg in legs)]
    if real:
        verdict, code = "RED", 1
    elif stale:
        verdict, code = "GREEN+STALE", 2
    else:
        verdict, code = "GREEN", 0
    return {"verdict": verdict, "code": code, "real": real,
            "excused": excused, "stale": stale}


def _line(label: str, legs: list, j: dict, head: str) -> str:
    parts = [f"GATE {head}", f"tier {label}"]
    for leg in legs:
        parts.append(f"{leg.name} {leg.summary} rc{leg.rc} ({leg.seconds:.0f}s)")
        td = "?" if leg.timing_deselected is None else str(leg.timing_deselected)
        parts.append(f"{leg.name}-timing-deselected {td}")
    parts.append(f"known-reds {len(j['excused'])}/{len(j['excused']) + len(j['stale'])}")
    parts.append(f"VERDICT {j['verdict']}")
    return " | ".join(parts)


def _report(label: str, legs: list, j: dict, head: str) -> None:
    text_by_leg = {leg.name: leg.failure_text for leg in legs}
    for leg in legs:
        print(f"  proves: {leg.name} = {leg.proves}")
    for row in j["excused"]:
        print(f"  known red (allowed): {row}")
    for e in j["stale"]:
        print(f"  STALE ALLOWLIST ENTRY: {e['id']} — allowed because {e['reason']!r} under "
              f"{e['condition']!r}, but it did not fire. {e['owner']} removes it.")
    for row in j["real"]:
        # `row` is "<leg name>:<nodeid>" (Leg names carry no ':'; a nodeid's own '::' is past
        # the first one), so split on the first ':' only. The assertion text travels here so
        # a red is diagnosable straight from this line, without opening the junit file or
        # re-running the test to find out what failed.
        leg_name, _, nodeid = row.partition(":")
        snippet = text_by_leg.get(leg_name, {}).get(nodeid)
        print(f"  RED: {row}")
        if snippet:
            print(f"        {snippet}")
    print(_line(label, legs, j, head))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Run TEX's gates and print one verdict. Exit 0 GREEN, 1 RED, "
                    "2 GREEN but the known-red allowlist is stale.")
    p.add_argument("--tier", choices=("cheap", "touched", "full"), default="cheap",
                   help="cheap = the eight ratchets; touched = the full test files that "
                        "import a module this branch touched, plus the docs/citation/"
                        "mutation/seam/skip-budget/floors ratchets, always -- a LANE's own "
                        "gate, run before handing back; full = cheap, then the CI shape and "
                        "the canonical whole-suite run (default: cheap)")
    p.add_argument("--base", default=None, metavar="REF",
                   help="`--tier touched` only: the ref to diff HEAD against when selecting "
                        "touched modules (default: origin/main)")
    p.add_argument("--no-cache", action="store_true",
                   help="ignore any cached verdict for this tree and tier, and refresh it")
    p.add_argument("--ci-python", default=None,
                   help=f"the interpreter for the CI shape (RUN only; never installed into). "
                        f"Falls back to ${_CI_PYTHON_ENV}, then to the interpreter running "
                        f"this script, which is said out loud because it proves less")
    p.add_argument("--python", default=sys.executable,
                   help="the interpreter for the cheap and canonical legs "
                        "(default: the one running this script)")
    p.add_argument("--counts-baseline", metavar="PATH",
                   help="add the structural-counts leg against this --save file, which must "
                        "have been taken at the gate shape: `host_path_counts.py --device cpu "
                        "--res 96 --window 48 --ticks 4 --prof1 off --save PATH`")
    p.add_argument("--keep-going", action="store_true",
                   help="run the full tier even when the cheap tier is red")
    p.add_argument("--scratch", metavar="DIR",
                   help="where per-leg TEX_CACHE_DIRs and junit files go (default: a temp dir)")
    p.add_argument("-v", "--verbose", action="store_true", help="echo each leg's output")
    a = p.parse_args(argv)

    if not _importable_as_tex_wrangle():
        print("the package must be importable as TEX_Wrangle: run from a directory where it "
              "resolves (e.g. a worktree whose package dir is named TEX_Wrangle)")
        return 2

    head, th = head_label(), tree_hash()
    ci_python, ci_source = resolve_ci_python(a.ci_python)
    base_ref = a.base or "origin/main"
    # Every interpreter THIS tier will run, in the key and on the line. The cheap tier never
    # touches the CI interpreter, so including it there would miss a cache hit for a question
    # that interpreter had no part in answering.
    interpreters = [("python", a.python)]
    if a.tier == "full":
        interpreters.append(("ci-python", ci_python))
    # `touched`'s selection depends on `base_ref` too (a diff against a different ref can pick
    # different files on an unchanged tree), so it rides the cache key -- folded into the tier
    # string rather than a new cache_key parameter, since it is the only tier this applies to.
    cache_tier = f"{a.tier}:{base_ref}" if a.tier == "touched" else a.tier
    key = cache_key(th, cache_tier, interpreters, bool(a.counts_baseline))
    who = describe_interpreters(interpreters)
    if not a.no_cache:
        hit = _cache_read(key)
        if hit:
            print(f"GATE {head} | tier {a.tier} | CACHED from {hit['at']} "
                  f"(tree and interpreters unchanged; --no-cache to re-run) | "
                  f"{hit.get('who', who)} | VERDICT {hit['verdict']}")
            for ln in hit.get("lines", []):
                print("  " + ln)
            return int(hit["code"])

    allowlist = load_allowlist()
    cuda = os.environ.get("CUDA_VISIBLE_DEVICES") != "-1"
    scratch = a.scratch or tempfile.mkdtemp(prefix="tex-gate-")
    os.makedirs(scratch, exist_ok=True)

    lines, codes = [], []
    print(f"  interpreters: {who}")

    if a.tier == "touched":
        # A standalone tier -- not layered on `cheap` (which the six ALWAYS ratchets already
        # partly overlap via `test_v017_phase2.py`); it is a lane's OWN gate, run instead of
        # (or beside) `cheap`, never a replacement for the orchestrator's landing `full`.
        touched = run_touched(a.python, base_ref, scratch, a.verbose)
        jt = judge([touched], allowlist, cuda)
        _report("touched", [touched], jt, head)
        lines.append(_line("touched", [touched], jt, head))
        codes.append(jt["code"])
    else:
        cheap = run_cheap(a.python, scratch, a.verbose)
        jc = judge([cheap], allowlist, cuda)
        _report("cheap", [cheap], jc, head)
        lines.append(_line("cheap", [cheap], jc, head))
        codes.append(jc["code"])

    if a.tier == "full":
        if jc["code"] == 1 and not a.keep_going:
            print(f"GATE {head} | tier full | SKIPPED (the cheap tier is red, and every "
                  f"cheap row re-runs inside the full tier) | VERDICT RED")
            print(f"GATE {head} | OVERALL RED")
            return 1
        full = [run_ci_shape(ci_python, scratch, a.verbose, ci_source),
                run_canonical(a.python, scratch, a.verbose)]
        if a.counts_baseline:
            full.append(run_counts(a.python, a.counts_baseline, scratch, a.verbose))
        jf = judge(full, allowlist, cuda)
        _report("full", full, jf, head)
        lines.append(_line("full", full, jf, head))
        codes.append(jf["code"])

    code = 1 if 1 in codes else (2 if 2 in codes else 0)
    overall = {0: "GREEN", 1: "RED", 2: "GREEN+STALE"}[code]
    at = datetime.datetime.now().replace(microsecond=0).isoformat(" ")
    final = f"GATE {head} | OVERALL {overall} | tree {th[:12]} | {at}"
    print(final)
    _cache_write(key, {"at": at, "verdict": overall, "code": code, "who": who,
                       "lines": lines + [final]})
    if not a.scratch:
        shutil.rmtree(scratch, ignore_errors=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
