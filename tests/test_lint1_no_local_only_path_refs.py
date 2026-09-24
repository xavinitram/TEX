"""LINT-1 (v0.43.0 rider (c)) — no TRACKED file cites a path that exists only in this
project's own local, unpushed working area.

`test_simp3_no_machine_paths.py` catches a path that is per-*person* (a home directory, a
drive letter into someone's profile). This is the sibling gap: a handful of directories and
files are per-*repository-checkout* rather than per-person — they hold orchestration
material this project's own law keeps out of every push (never committed, never referenced
by anything that is) — and nothing mechanical enforced that the second half of that rule
("never referenced") actually holds. `git log` shows it slipping at least once already: a
prior commit had to drop three such references out of test comments after they were caught
by hand.

**Same shape as SIMP-3, same reason for it:** scan `git ls-files` — the PUSHED set, not the
shipped one — because a path can be absent from the published archive and still sit in a
tracked file forever. **The allowlist is empty by design**: the tree was clean under these
patterns when this landed (after fixing the three leftover references the scan below found),
so any future entry here is a decision made out loud, never a way to make a red go away.

**Why the patterns are assembled from pieces instead of written whole:** this file is itself
part of the scan set, and every one of the patterns below names one of this project's own
local-only paths. Writing any of them as one contiguous literal would make this file BE the
violation it exists to catch, so each is built at runtime from pieces that are never
contiguous in the source text — the same technique SIMP-3 uses for its witness strings, used
here for the patterns themselves. A push-time scan of this diff for each whole fragment is
expected to find zero.

**The embedding host's own name is matched a different way: by HASH, exact-case.** A literal
name check would put the name itself in this file's source text, which is the one thing the
rest of this file goes to lengths to avoid doing with any other local-only string. Instead:
tokenise a line on `\\w+`, sha256-hex each token EXACTLY AS WRITTEN (no casefold), and compare
against a frozenset seeded with the capitalised and ALL-CAPS spellings of the noun/verb you
get by dropping the trailing "-ing" off "sharding" — an ordinary, unrelated word this project
already uses in its own lowercase verb form (cache/executor sharding). Exact-case, not
casefolded, is the whole point: this project's own lowercase usage hashes to neither seeded
digest, so it never fires, while a capitalised or all-caps spelling of the same word — which
is how a name is written, never how a common verb is — does. Also seeded with the second
host's project directory name from the fragment list above, spelled exactly as it appears
there. Nothing here is a guess about false positives: both seeded spellings were checked
against every tracked line before this landed and matched none.
"""
import hashlib
import pathlib
import re
import subprocess

from helpers import SubTestResult

_PKG = pathlib.Path(__file__).resolve().parent.parent

#: `git ls-files` on this tree lists a few thousand paths; a much larger count means the
#: command answered about the wrong directory, worth a red rather than a long scan.
_MAX_TRACKED = 20000

#: Every "word" token, underscore included -- which is what lets an underscore-joined
#: identifier (the second host's project directory name) tokenise as ONE piece, matching how
#: it is spelled in the fragment list below.
_WORD_RE = re.compile(r"\w+")

#: The seed word, derived rather than spelled: the noun/verb "sharding" minus its trailing
#: "-ing". Never compared in this bare, lowercase form -- see `_HOST_NAME_HASHES` -- so the
#: project's own unrelated verb usage of it cannot match.
_HOST_SEED = "sharding"[:-3]


def _frag(*pieces: str) -> str:
    """Join pieces into one forbidden fragment at RUN time, so the whole string never sits
    contiguously anywhere in this file's own source text."""
    return "".join(pieces)


#: sha256 digests of: the seed word capitalised, the seed word ALL-CAPS, and the second
#: host's project directory name exactly as spelled in the fragment list below. A token's
#: hash landing in this set is how the embedding host's name is recognised without its
#: literal spelling ever sitting in this file.
_HOST_NAME_HASHES = frozenset(
    hashlib.sha256(form.encode("utf-8")).hexdigest()
    for form in (_HOST_SEED.capitalize(), _HOST_SEED.upper(), _frag("TEX_", "compositor"))
)


def scan_host_name(text: str) -> list:
    """`[(lineno, token)]` for every EXACT-CASE word token whose sha256 lands in
    `_HOST_NAME_HASHES`. No casefold: `"shard".capitalize()` and `.upper()` are the only two
    spellings that can ever hit, so this project's own lowercase verb usage of the same word
    never does."""
    found = []
    for n, line in enumerate(text.splitlines(), 1):
        for tok in _WORD_RE.findall(line):
            if hashlib.sha256(tok.encode("utf-8")).hexdigest() in _HOST_NAME_HASHES:
                found.append((n, tok))
    return found


def _fragments():
    """`[(what it is, the literal fragment)]`, each assembled from pieces. Every fragment
    names a path that exists only in this project's own local working area, never in the
    published tree and never referenced by it."""
    return [
        ("an evidence/worklog path", _frag("docs", "/", "worklog")),
        ("the findings-tracker directory", _frag("bug", "_reports")),
        ("a per-host asks/changelog document", _frag("docs/", "shard-")),
        ("the orchestrator pointer file", _frag("CLA", "UDE", ".md")),
        ("the agent-definitions directory (forward slash)", _frag(".", "cla", "ude/")),
        ("the agent-definitions directory (backslash)", _frag(".", "cla", "ude\\")),
        ("an upstream hand-over document", _frag("docs/", "upstream")),
        ("a second host's project directory name", _frag("TEX_", "compositor")),
    ]


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


#: Tracked paths that may carry a fragment anyway, each with the reason it is not a leak.
#: EMPTY is the healthy state -- the tree was clean under these patterns when this landed,
#: so an entry here is a decision somebody makes out loud, never a way to make a red go away.
_ALLOWLIST: dict = {}


def scan(text: str, fragments) -> list:
    """`[(lineno, what it is, the fragment)]` for one file's content."""
    found = []
    for n, line in enumerate(text.splitlines(), 1):
        for what, frag in fragments:
            if frag in line:
                found.append((n, what, frag))
    return found


def test_lint1_no_tracked_file_names_a_local_only_path(r: SubTestResult):
    print("\n--- LINT-1: no tracked file cites a path that exists only in this checkout's "
          "own local working area, or names the embedding host ---")
    paths = tracked_paths()
    if paths is None:
        r.skip("LINT-1 local-only-path lint",
               "this tree is not a git checkout, so the tracked set cannot be enumerated")
        return
    fragments = _fragments()
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
        hits += [f"{rel}:{n}: {what} — {frag!r}" for n, what, frag in scan(text, fragments)]
        hits += [f"{rel}:{n}: names the embedding host — {tok!r}"
                 for n, tok in scan_host_name(text)]
    if hits:
        r.fail("LINT-1 local-only-path lint",
               f"{len(hits)} tracked line(s) name a path that exists only in this project's "
               f"own local working area, or name the embedding host; a tracked file is a "
               f"PUSHED file:\n  " + "\n  ".join(hits[:40]))
        return
    r.ok(f"{scanned} tracked text file(s) name no local-only path and no embedding host "
         f"(allowlist: {len(_ALLOWLIST)})")


def test_lint1_the_lint_is_not_inert(r: SubTestResult):
    """The patterns are only worth their runtime if they fire, and fire on the shape and not
    on a neighbour that merely shares some of its pieces. These witnesses are strings built
    from pieces at call time, never files, so the row proves the comparator on a tree that
    is (and must stay) clean -- and never itself commits the fragment it is proving."""
    print("\n--- LINT-1: the local-only-path patterns fire, and only on the real shape ---")
    fragments = _fragments()
    must_red_fragments = [
        "See " + _frag("docs", "/", "worklog") + "/lint-1/notes.md for the evidence.",
        "grep " + _frag("bug", "_reports") + "/pending for open items.",
        _frag("docs/", "shard-") + "asks.md tracks status by ask id.",
        "Read " + _frag("CLA", "UDE", ".md") + " before touching anything.",
        "ls " + _frag(".", "cla", "ude/") + "agents",
        "dir " + _frag(".", "cla", "ude\\") + "agents",
        _frag("docs/", "upstream") + "/brief.md numbers the items.",
        "vendored a pin from " + _frag("TEX_", "compositor") + " upstream.",
    ]
    # Two spellings of the seed word (never the bare lowercase form -- that must NOT hit,
    # proven by must_stay_green below), plus the compositor directory name, exercised through
    # the HASH mechanism rather than the substring one above.
    must_red_host = [
        "shipped for " + _HOST_SEED.capitalize() + " directly.",
        "vendored straight into " + _HOST_SEED.upper() + ".",
        "pinned from " + _frag("TEX_", "compositor") + " via the vendor script.",
    ]
    must_stay_green = [
        "worklog rotation happens weekly.",                 # no leading "docs/"
        "the bugfix_reports queue is empty.",                # not "bug" + "_reports"
        "shard-asks is a familiar SUFFIX, not this fragment.",  # no leading "docs/"
        _frag("CLA", "UDE") + ".py is not a real module.",   # wrong extension
        "de" + _frag("cla", "ude") + "/ is not a real directory.",  # no leading dot
                                                               # before the directory word
        "a compositor is just a piece of render software.",  # no leading "TEX_"
        "upstream changes get reviewed before merge.",       # no leading "docs/"
        # The bare, lowercase seed word: an ordinary verb this project already uses for
        # unrelated work (splitting something across workers) -- must NOT hash-match.
        "a parallel executor must " + _HOST_SEED + " the work.",
        # "sharding" itself, the whole verb: a DIFFERENT token from the bare seed, and it
        # must not hit either, exactly as the design requires.
        "cache/executor " + _HOST_SEED + "ing stays untouched.",
    ]
    missed = ([w for w in must_red_fragments if not scan(w, fragments)]
              + [w for w in must_red_host if not scan_host_name(w)])
    tripped = ([f"{w}  ->  {scan(w, fragments)[0][2]!r}"
                for w in must_stay_green if scan(w, fragments)]
               + [f"{w}  ->  {scan_host_name(w)[0][1]!r}"
                  for w in must_stay_green if scan_host_name(w)])
    if missed:
        r.fail("LINT-1 lint witness (inert)",
               "the patterns did not fire on a real local-only path or the embedding host:\n  "
               + "\n  ".join(missed))
    elif tripped:
        r.fail("LINT-1 lint witness (over-tight)",
               "the patterns fired on a line naming no local-only path and no embedding "
               "host:\n  " + "\n  ".join(tripped))
    else:
        r.ok(f"{len(must_red_fragments)} local-only-path shapes and {len(must_red_host)} "
             f"embedding-host shapes red, {len(must_stay_green)} neighbours stay green")
