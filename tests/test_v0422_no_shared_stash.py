"""v0422-gatehyg — TRK-128's rule, made mechanical: `refs/stash` is forbidden here.

`refs/stash` lives in the ONE shared `.git` directory of a repository, so every
`git worktree` checkout of this repository shares one stash stack. `TRK-128`
(the project's findings record): two lanes stashed at essentially the same moment and each
`pop` recovered the OTHER lane's entry, git reporting nothing wrong at either step in
either worktree. The fix that landed was a positive instruction to every implementer
("NEVER `git stash`") -- correct, but an instruction is not a gate, and the fast tier
had no mechanical check of its own for the one condition that makes the hazard live: a
stash actually sitting in `refs/stash` right now, left by whichever lane wrote it last.

This is that mechanical check. It fails loudly, naming the rule and the two ways to set
changes aside instead, and it never itself leaves a stash in THIS repository to prove
the check works -- that would be the exact violation this file exists to catch, so the
self-test below runs against a throwaway temporary repository instead.
"""
import pathlib
import shutil
import subprocess
import tempfile

from helpers import SubTestResult

_PKG = pathlib.Path(__file__).resolve().parent.parent


def _has_stash(repo_dir) -> bool:
    """True if `refs/stash` resolves inside the given git repository/worktree."""
    proc = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "--verify", "--quiet", "refs/stash"],
        capture_output=True, text=True)
    return proc.returncode == 0


def test_v0422_no_shared_stash_in_this_repository(r: SubTestResult):
    print("\n--- v0422-gatehyg / TRK-128: refs/stash is forbidden here (mechanical) ---")
    try:
        if _has_stash(_PKG):
            r.fail(
                "no shared stash",
                "refs/stash exists in this repository. git's stash is ONE stack shared "
                "by every `git worktree` checkout of this repository (TRK-128: two lanes "
                "each silently received the other's uncommitted diff) -- project law "
                "forbids using it here. Set your changes aside instead: `git diff > "
                "<a scratch path outside the repo>` and reapply with `git apply`, or "
                "commit them to your own branch. Then `git stash drop` before re-running "
                "this gate.")
            return
        r.ok("no refs/stash in this repository")
    except Exception as e:
        r.fail("no shared stash", f"{type(e).__name__}: {e}")


def test_v0422_stash_checker_detects_a_real_stash(r: SubTestResult):
    print("\n--- v0422-gatehyg: the checker itself, proven on a throwaway repo ---")
    # Never exercised against THIS repository -- creating a stash here to prove the
    # checker works would commit the exact violation the row above exists to catch.
    tmp = tempfile.mkdtemp(prefix="tex-stash-check-")
    try:
        def run(*args):
            return subprocess.run(["git", "-C", tmp, *args], capture_output=True,
                                  text=True, check=True)
        run("init", "-q")
        run("config", "user.email", "tex-gate-hygiene@example.invalid")
        run("config", "user.name", "tex-gate-hygiene")
        f = pathlib.Path(tmp) / "f.txt"
        f.write_text("one\n", encoding="utf-8")
        run("add", "f.txt")
        run("commit", "-q", "-m", "seed")
        assert not _has_stash(tmp), "a fresh repo must never read as already stashed"
        f.write_text("two\n", encoding="utf-8")
        run("stash", "-q")
        assert _has_stash(tmp), "the checker must detect a real refs/stash"
        r.ok("checker: absent -> False, present -> True, on a throwaway repo")
    except Exception as e:
        r.fail("stash checker self-test", f"{type(e).__name__}: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
