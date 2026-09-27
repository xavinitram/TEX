#!/usr/bin/env python3
"""scan_commit_messages.py -- G5 (FIX-GATE, v0.47.0 Phase C, R4#4).

`tests/test_lint1_no_local_only_path_refs.py` (LINT-1) scans tracked FILE CONTENT for a
local-only path, the embedding host's name, or a budgeted-over bare word -- it has never
scanned commit MESSAGES, because `git ls-files` only ever walks the tree. That is the exact
gap one AUTO-47 lane's own local write-up hit: an earlier head's commit MESSAGE named this
project's second host, caught only by a one-off manual grep that lane happened to run on
its own initiative, with the cheap tier fully GREEN the whole time. This is the standing
version of that grep, meant to run once per landing, over the commit range about to be
pushed.

WHY IT REUSES `tests/test_lint1_no_local_only_path_refs.py` RATHER THAN ITS OWN COPY: the
fragment list, the host-name hash set and the (G5) bare-word hash set are exactly what
counts as a leak, and that list must never drift between "what LINT-1 checks in a file" and
"what this checks in a commit message" -- so this module imports that one, it does not
re-derive it (the reuse gap R1 flagged elsewhere in this round, avoided here on purpose).

WHAT COUNTS AS A HIT, per commit message (subject + body, `%B`):
  * any local-only path fragment (`_fragments()`), the embedding host's name
    (`scan_host_name`), OR a bare word from the G5 hash set (`scan_bare_words`) --
    UNBUDGETED here: a commit message is never grandfathered the way a pre-existing
    source comment can be, because nothing legitimate requires naming any of these in a
    message that gets pushed.

USAGE
-----
    python tools/scan_commit_messages.py BASE_SHA..HEAD_SHA
    python tools/scan_commit_messages.py BASE_SHA HEAD_SHA

Exit 0: no commit in the range leaks anything. Exit 1: at least one does (each hit is
printed as `<short sha>: <what> -- <token/fragment>`). Exit 2: bad invocation.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent.parent

#: A separator byte sequence that cannot appear inside a commit message, so one `git log`
#: call can be split back into (sha, message) pairs without a second process per commit.
_REC_SEP = "\x1e"
_UNIT_SEP = "\x1f"


def _git_log_messages(rev_range: str, cwd: Path = _PKG) -> list[tuple[str, str]]:
    """`[(full sha, subject+body)]` for every commit in `rev_range`, oldest first."""
    fmt = f"%H{_UNIT_SEP}%B{_REC_SEP}"
    out = subprocess.run(["git", "log", rev_range, f"--format={fmt}"],
                         cwd=str(cwd), capture_output=True, text=True, check=True).stdout
    commits = []
    for rec in out.split(_REC_SEP):
        rec = rec.strip("\n")
        if not rec:
            continue
        sha, _, body = rec.partition(_UNIT_SEP)
        commits.append((sha, body))
    return commits


def _load_lint1(pkg: Path = _PKG):
    """Import the maintained fragment/hash definitions straight from the test file that
    owns them, rather than keeping a second copy here that could quietly drift from what
    LINT-1 actually checks in a tracked file."""
    tests_dir = str(pkg / "tests")
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    import test_lint1_no_local_only_path_refs as lint1
    return lint1


def scan_range(rev_range: str, repo_dir: Path | None = None, pkg: Path = _PKG) -> list[str]:
    """`[<short sha>: <what> -- <token/fragment>]` for every leak found in `rev_range`.

    `repo_dir` is where `git log` runs (defaults to the process's OWN current directory,
    matching every other `git`-invoking entry point in `tools/`, e.g. `gate.py`'s `_git`);
    `pkg` is always where the shared fragment/hash definitions are imported from, regardless
    of `repo_dir` -- kept separate so a test can point `repo_dir` at a throwaway repository
    while still exercising this project's own real, maintained leak definitions rather than
    a copy."""
    repo_dir = repo_dir if repo_dir is not None else Path.cwd()
    lint1 = _load_lint1(pkg)
    fragments = lint1._fragments()
    hits: list[str] = []
    for sha, body in _git_log_messages(rev_range, repo_dir):
        short = sha[:10]
        for _n, what, frag in lint1.scan(body, fragments):
            hits.append(f"{short}: {what} -- {frag!r}")
        for _n, tok in lint1.scan_host_name(body):
            hits.append(f"{short}: names the embedding host -- {tok!r}")
        for _n, tok in lint1.scan_bare_words(body):
            hits.append(f"{short}: names a leak-class bare word (never budgeted in a "
                        f"commit message) -- {tok!r}")
    return hits


def main(argv: list | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 1:
        rev_range = argv[0]
    elif len(argv) == 2:
        rev_range = f"{argv[0]}..{argv[1]}"
    else:
        print(__doc__)
        return 2
    hits = scan_range(rev_range)
    if hits:
        print(f"{len(hits)} commit message leak(s) found in {rev_range}:")
        for h in hits:
            print(f"  {h}")
        return 1
    print(f"0 commit message leaks found in {rev_range}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
