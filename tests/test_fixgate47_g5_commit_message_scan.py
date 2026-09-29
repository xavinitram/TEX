"""FIX-GATE G5 (v0.47.0 Phase C, R4#4) -- `tools/scan_commit_messages.py`.

LINT-1 (`tests/test_lint1_no_local_only_path_refs.py`) walks `git ls-files` -- tree
content -- and has never looked at commit MESSAGES, which is exactly the gap one AUTO-47
lane's own local write-up hit: an earlier head's commit message named this project's second
host, caught only by a one-off manual grep, with the cheap tier fully green throughout.
This pins the standing replacement: a small throwaway git repository stands in for "the
range about to land," so the test never depends on -- or risks polluting -- this
repository's own real history.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

from helpers import SubTestResult
from helpers import scratch_dir

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import scan_commit_messages as SCM   # noqa: E402


def _git(repo: Path, *args):
    # The throwaway repo must not inherit the developer's global git config (commit signing).
    return subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=str(repo),
                          capture_output=True, text=True, check=True)


def _mini_repo() -> Path:
    d = scratch_dir("tex_g5_commitscan_")
    _git(d, "init", "-q", "--template=")
    _git(d, "config", "user.email", "test@example.invalid")
    _git(d, "config", "user.name", "test")
    (d / "a.txt").write_text("hello\n", encoding="utf-8")
    _git(d, "add", "a.txt")
    _git(d, "commit", "-q", "-m", "base: initial commit")
    return d


def test_fixgate_g5_clean_range_scans_zero(r: SubTestResult):
    print("\n--- G5: a clean commit range scans to zero hits ---")
    try:
        d = _mini_repo()
        (d / "a.txt").write_text("hello again\n", encoding="utf-8")
        _git(d, "add", "a.txt")
        _git(d, "commit", "-q", "-m", "TIDY-1: reword the greeting\n\nComfyUI-invisible.")
        base = _git(d, "rev-list", "--max-parents=0", "HEAD").stdout.strip()
        hits = SCM.scan_range(f"{base}..HEAD", repo_dir=d)
        assert hits == [], hits
        r.ok("a clean commit range reports zero leaks")
    except Exception as e:
        r.fail("G5 clean range", str(e))


def test_fixgate_g5_a_local_only_path_fragment_in_a_commit_message_is_caught(r: SubTestResult):
    print("\n--- G5: a local-only path fragment in a commit MESSAGE is caught ---")
    try:
        d = _mini_repo()
        base = _git(d, "rev-parse", "HEAD").stdout.strip()
        (d / "a.txt").write_text("hello once more\n", encoding="utf-8")
        _git(d, "add", "a.txt")
        # Built from pieces, never spelled contiguously in THIS file either -- same
        # discipline LINT-1 itself follows, exercised here as a commit message instead of
        # tracked file content.
        leak = "see " + "docs" + "/" + "worklog" + "/notes.md"
        _git(d, "commit", "-q", "-m", f"OOPS-1: {leak}")
        hits = SCM.scan_range(f"{base}..HEAD", repo_dir=d)
        assert len(hits) == 1 and "evidence/worklog path" in hits[0], hits
        r.ok(f"a local-only path fragment in a commit message is caught: {hits}")
    except Exception as e:
        r.fail("G5 local-only path in commit message", str(e))


def test_fixgate_g5_a_bare_word_in_a_commit_message_is_caught_unbudgeted(r: SubTestResult):
    print("\n--- G5: a bare word in a commit MESSAGE is caught with NO budget ---")
    try:
        d = _mini_repo()
        base = _git(d, "rev-parse", "HEAD").stdout.strip()
        (d / "a.txt").write_text("hello a third time\n", encoding="utf-8")
        _git(d, "add", "a.txt")
        leak = "hand" + "-back"   # never spelled contiguously here either
        _git(d, "commit", "-q", "-m", f"OOPS-2: see the {leak} for details")
        hits = SCM.scan_range(f"{base}..HEAD", repo_dir=d)
        assert len(hits) == 1 and "bare word" in hits[0], hits
        r.ok(f"a bare-word leak in a commit message is caught: {hits}")
    except Exception as e:
        r.fail("G5 bare word in commit message", str(e))


def test_fixgate_g5_cli_exit_codes(r: SubTestResult):
    print("\n--- G5: the CLI's exit code is 0 clean, 1 dirty ---")
    try:
        d = _mini_repo()
        base = _git(d, "rev-parse", "HEAD").stdout.strip()
        (d / "a.txt").write_text("clean change\n", encoding="utf-8")
        _git(d, "add", "a.txt")
        _git(d, "commit", "-q", "-m", "CLEAN-1: an ordinary change")
        script = str(Path(__file__).resolve().parent.parent / "tools" / "scan_commit_messages.py")
        clean = subprocess.run([sys.executable, script, f"{base}..HEAD"], cwd=str(d),
                               capture_output=True, text=True)
        assert clean.returncode == 0, clean.stdout + clean.stderr

        leak = "docs" + "/" + "upstream"
        (d / "a.txt").write_text("dirty change\n", encoding="utf-8")
        _git(d, "add", "a.txt")
        _git(d, "commit", "-q", "-m", f"OOPS-3: read {leak}/brief.md first")
        dirty = subprocess.run([sys.executable, script, f"HEAD~1..HEAD"], cwd=str(d),
                               capture_output=True, text=True)
        assert dirty.returncode == 1, dirty.stdout + dirty.stderr
        r.ok("CLI exits 0 on a clean range and 1 on a leaking one")
    except Exception as e:
        r.fail("G5 CLI exit codes", str(e))
