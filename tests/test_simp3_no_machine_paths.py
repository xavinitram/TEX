"""No TRACKED file names one person's home, profile or project root.

Two standing rules had no enforcer behind them: nothing pushed names a particular machine's
private layout, and, more generally, no tracked file carries an absolute path that only exists
on the box it was written on. The second implies the first and is the mechanical one.

It was not hypothetical. A tooling change shipped a hard-coded absolute path to a second Python
interpreter under the author's own project root, and EVERY gate passed — the CI shape, the
canonical run, the archive-surface ratchet and the doc registries. It was caught by a human
reading the diff, which is a check that works until the week nobody runs it.

Why every ratchet missed it is the part worth keeping: the leak lived in `tools/`, which
`.comfyignore` excludes, so the shipped-surface ratchets deliberately never look there. *Does
not ship* and *is not pushed* are different sets, and they had drifted apart without anybody
noticing. So this lint scans what `git ls-files` reports — the PUSHED set — and not the shipped
one.

**The allowlist is empty, and that is the whole point.** The tree was clean under these patterns
when this landed, so any entry ever added here is a decision somebody makes out loud.

**What it matches, and what it deliberately does not.** A path is flagged when its ROOT is a
per-person one: a drive-letter path into a user-profile or project root, a home directory under
the POSIX home or user root, either of those reached through a single-letter drive directory at
the shell's root (with or without an `/mnt` prefix), or a drive-letter path into an embedding
host's install directory. Those roots differ per machine by construction, which is what makes
the match a fact rather than a guess — nothing fixes where somebody unpacked a host, any more
than it fixes what their username is.

It does NOT flag every drive-letter path, because absolute is not the same as machine-specific:
a search pattern under `Program Files` names a location every Windows box has, and a synthetic
drive in a test fixture names no box at all. Measured at head, the broad "any drive letter" rule
reds on eight legitimate lines and one escape sequence for every one real leak, so it would
arrive with an allowlist on its first day — and an allowlist written to make a new gate go green
teaches the next reader to add to it. It also does not flag a bare directory NAME: a host's
install folder is named in a comment and in a usage line at head, and a name is only a leak once
it sits inside an absolute path, which the patterns above already catch. Both of those tracked
lines are witnesses in the row below, so the carve-out cannot quietly widen either.

A URL is not a path: the drive-letter patterns require the letter to stand alone, so `https://`
cannot match. This file is inside its own scan set, so every pattern is written to describe the
shape without containing an instance of it — a lint that had to exempt itself would have a hole
exactly where the example lives.
"""
import pathlib
import re
import subprocess

from helpers import SubTestResult

_PKG = pathlib.Path(__file__).resolve().parent.parent

#: (what makes it per-person, the pattern). A drive letter must stand ALONE — preceded by
#: something that is not a word character — so no URL scheme can match it.
_PATTERNS = [
    ("a drive-letter path into a per-person root",
     re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]{1,2}(?:Users|Documents and Settings|Projects)"
                r"[\\/]", re.I)),
    ("a home directory",
     re.compile(r"/(?:home|Users)/[A-Za-z0-9._-]+/")),
    ("a POSIX spelling of a Windows user-profile root",
     re.compile(r"(?<![A-Za-z0-9_])(?:/mnt)?/[a-z]/Users/", re.I)),
    # An embedding host's install directory is a per-machine root for the same reason a
    # profile directory is: nothing fixes where somebody unpacked it. This is the pattern
    # that would have caught the benchmark fallback fixed alongside it - a literal
    # `G:\<somebody>\comfyUI\custom_nodes` sitting in a tracked file since v0.16.
    ("a drive-letter path into an embedding host's install",
     re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]{1,2}[^\s\"'<>|]*[Cc]omfy", re.I)),
]

#: Tracked paths that may carry one anyway, each with the reason it is not a leak. EMPTY is the
#: healthy state: the tree was clean when this row landed, so an entry here is a decision, never
#: a way to make a red go away.
_ALLOWLIST: dict = {}

#: `git ls-files` on this tree lists a few thousand paths; anything wildly beyond that means the
#: command answered about the wrong directory, which is worth a red rather than a long scan.
_MAX_TRACKED = 20000


def tracked_paths():
    """Every tracked path, or `None` when this tree is not a git checkout."""
    try:
        out = subprocess.run(["git", "ls-files", "-z"], cwd=str(_PKG), capture_output=True,
                             text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return [p for p in out.stdout.split("\0") if p][:_MAX_TRACKED]


def _spell(sep: str, *segments) -> str:
    """Build a witness path from its segments, so no literal instance sits in this source."""
    return sep.join(segments)


def scan(rel: str, text: str) -> list:
    """`[(lineno, why, matched text)]` for one file's content."""
    found = []
    for n, line in enumerate(text.splitlines(), 1):
        for why, pat in _PATTERNS:
            m = pat.search(line)
            if m:
                found.append((n, why, m.group(0)))
    return found


def test_simp3_no_tracked_file_names_a_private_root(r: SubTestResult):
    print("\n--- SIMP-3: no tracked file names a home, profile or project root ---")
    paths = tracked_paths()
    if paths is None:
        r.skip("SIMP-3 private-root lint",
               "this tree is not a git checkout, so the tracked set cannot be enumerated")
        return
    hits, scanned = [], 0
    for rel in paths:
        if rel in _ALLOWLIST:
            continue
        p = _PKG / rel
        if not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue                      # binary, or gone since `ls-files` answered
        scanned += 1
        hits += [f"{rel}:{n}: {why} — {m!r}" for n, why, m in scan(rel, text)]
    if hits:
        r.fail("SIMP-3 private-root lint",
               f"{len(hits)} tracked line(s) name a root that exists on one box only; a "
               f"tracked file is a PUSHED file, whatever `.comfyignore` says about it:\n  "
               + "\n  ".join(hits[:40]))
        return
    r.ok(f"{scanned} tracked text file(s) name no private root "
         f"(allowlist: {len(_ALLOWLIST)})")


def test_simp3_the_private_root_lint_is_not_inert(r: SubTestResult):
    """The patterns are only worth their runtime if they fire — and only if they fire on the
    shape and not on its neighbours. These witnesses are strings, not files, so the row proves
    the comparator on a tree that is (and must stay) clean.

    Every witness is SPELLED from its segments rather than written out, because this file is
    inside the scan set: a literal example here would be a tracked line naming a private root,
    and the lint would red on its own examples. (It did, the hour it was first committed, which
    is as good a demonstration that it works as any planted probe.)
    """
    print("\n--- SIMP-3: the private-root patterns fire, and only on a private root ---")
    host = "Comfy" + "UI_windows_portable"
    must_red = [
        "PY = '" + _spell("\\", "C:", "Users", "someone", "python.exe") + "'",
        "PY = '" + _spell("/", "D:", "Projects", "a_tree", ".venv", "python.exe") + "'",
        "run('" + _spell("/", "", "home", "someone", "build.sh") + "')",
        "PY = '" + _spell("/", "", "c", "Users", "someone", "python.exe") + "'",
        "PY = '" + _spell("/", "", "mnt", "d", "Users", "someone", "tool") + "'",
        "PY = r'" + _spell("\\", "C:", host, "python_embeded", "python.exe") + "'",
        "_CN = r'" + _spell("\\", "G:", "Somebody_Menu", "comfyUI", "custom_nodes") + "'",
    ]
    must_stay_green = [
        "url = 'https://example.invalid/a'",
        "    'if not k:\\n'",
        "pattern = r'C" + ":" + "\\\\Program Files\\\\Vendor\\\\**\\\\tool.bat'",
        "os.environ['CACHE_DIR'] = 'Z" + ":" + "/a_fixture_value'",
        "See tools/gate.py for the tier list.",
        # The host's directory NAME is not a private root; only an absolute path into it
        # is. Both of these are tracked lines at head and both must stay green.
        "Run:  python_embeded/python.exe -X utf8 benchmarks/lat4_ab.py",
        '# named "' + host + '" too, so a naive filter would take torch out with it).',
    ]
    missed = [w for w in must_red if not scan("w", w)]
    tripped = [f"{w}  ->  {scan('w', w)[0][2]!r}" for w in must_stay_green if scan("w", w)]
    if missed:
        r.fail("SIMP-3 lint witness (inert)",
               "the patterns did not fire on a private root:\n  " + "\n  ".join(missed))
    elif tripped:
        r.fail("SIMP-3 lint witness (over-tight)",
               "the patterns fired on a line that names no box:\n  " + "\n  ".join(tripped))
    else:
        r.ok(f"{len(must_red)} private-root shapes red, {len(must_stay_green)} neighbours stay "
             f"green")
