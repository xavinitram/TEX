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
`--tier cheap` runs the six ratchets that answer in seconds: the no-numpy ban, the LOC and
headroom ratchets, the archive-surface ratchet, the host-path counts pins, TST-7's runner
drift check, and the private-root lint over the tracked set. Every one of them is a strict
SUBSET of the full tier; they are kept for feedback latency, not for coverage, and this tool
says so out loud.

`--tier full` runs cheap first (cheapest first, and it aborts there if cheap is red unless
`--keep-going`), then the two whole-suite legs that are NOT subsets of each other:

  * **ci-shape** — a second interpreter, ideally the Python version CI uses and one with no
    embedding host installed, run from the package ROOT so the host is off `sys.path`, with
    `CUDA_VISIBLE_DEVICES=-1`, `-m "not slow"`, `-p no:cacheprovider`. It is the only leg that
    can catch a test which assumes a host or a GPU. It runs on whatever OS you are on, so it
    cannot catch a line-ending or toolchain difference — say that when quoting it.

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

`tools/` is excluded from the published archive (`.comfyignore`), so nothing here ships.
"""
from __future__ import annotations

import argparse
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
]

#: Where the CI-shape interpreter is named, so this file names no machine's private layout.
_CI_PYTHON_ENV = "TEX_CI_PYTHON"

_SUMMARY_RE = re.compile(
    r"^[=\s]*\d+ (?:passed|failed|error|deselected|skipped)|"
    r"^\s*(?:\d+ \w+,? ?)+ in [\d.]+s", re.I)


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


def _run(leg: Leg, argv: list, cwd: str, env_extra: dict, scratch: str, verbose: bool) -> Leg:
    cache = os.path.join(scratch, f"cache-{leg.name}")
    shutil.rmtree(cache, ignore_errors=True)
    os.makedirs(cache, exist_ok=True)
    junit = os.path.join(scratch, f"junit-{leg.name}.xml")
    env = dict(os.environ, TEX_CACHE_DIR=cache, **env_extra)
    t0 = time.time()
    proc = subprocess.run(argv + [f"--junit-xml={junit}"], cwd=cwd, env=env,
                          capture_output=True, text=True, errors="replace")
    leg.seconds = time.time() - t0
    leg.rc = proc.returncode
    leg.summary = _summary_of(proc.stdout)
    leg.failures, leg.collected, leg.failure_text = _parse_junit(junit)
    if verbose:
        print(f"\n--- {leg.name}: {' '.join(argv)} (cwd={cwd}) ---")
        print(proc.stdout[-8000:])
        if proc.stderr.strip():
            print(proc.stderr[-2000:])
    return leg


def run_cheap(python: str, scratch: str, verbose: bool) -> Leg:
    leg = Leg("cheap", "the six ratchets only — no whole-suite collection, "
                       "no host-absent lane, CUDA present")
    files = [f"TEX_Wrangle/{p}" for _, p in _CHEAP]
    argv = [python, "-X", "utf8", _HARNESS, *files, "-q", "-p", "no:cacheprovider"]
    return _run(leg, argv, _PARENT, {}, scratch, verbose)


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
    argv = [ci_python, "-m", "pytest", "tests/", "-q", "-m", "not slow",
            "-p", "no:cacheprovider"]
    return _run(leg, argv, _PKG, {"CUDA_VISIBLE_DEVICES": "-1"}, scratch, verbose)


def run_canonical(python: str, scratch: str, verbose: bool) -> Leg:
    leg = Leg("canonical", "the embedded interpreter with CUDA and the host present, the v3 "
                           "NodeOutput wrapper disarmed — the only leg that runs the GPU rows")
    argv = [python, "-X", "utf8", _HARNESS, "TEX_Wrangle/tests", "-q", "-m", "not slow",
            "-p", "no:cacheprovider"]
    return _run(leg, argv, _PARENT, {}, scratch, verbose)


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
    proc = subprocess.run(argv, cwd=_PARENT, env=dict(os.environ, TEX_CACHE_DIR=cache),
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
            # `judge()` only ever looks at `leg.failures`, never `leg.rc`, so a leg like this
            # used to fall through to GREEN with nothing to red on. Give it a synthetic id no
            # allowlist entry can name, so it always lands in `real` instead.
            infra_id = f"<counts:infra-rc{leg.rc}>"
            leg.failures = [infra_id]
            leg.failure_text[infra_id] = (tail[-1] if tail else
                                          proc.stdout.strip().splitlines()[-1]
                                          if proc.stdout.strip() else "(no output)")
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
    p.add_argument("--tier", choices=("cheap", "full"), default="cheap",
                   help="cheap = the six ratchets; full = cheap, then the CI shape and the "
                        "canonical whole-suite run (default: cheap)")
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

    head, th = head_label(), tree_hash()
    ci_python, ci_source = resolve_ci_python(a.ci_python)
    # Every interpreter THIS tier will run, in the key and on the line. The cheap tier never
    # touches the CI interpreter, so including it there would miss a cache hit for a question
    # that interpreter had no part in answering.
    interpreters = [("python", a.python)]
    if a.tier == "full":
        interpreters.append(("ci-python", ci_python))
    key = cache_key(th, a.tier, interpreters, bool(a.counts_baseline))
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
