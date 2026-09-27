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

from helpers import SubTestResult
from test_simp3_no_machine_paths import tracked_paths   # same PUSHED-set walk SIMP-3 already
                                                          # does; no second implementation here.

_PKG = pathlib.Path(__file__).resolve().parent.parent

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
    `_HOST_NAME_HASHES`. No casefold: the seed word's own two cased spellings (see
    `_HOST_SEED`, above) are the only ones that can ever hit, so this project's own lowercase
    verb usage of the same word never does."""
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
        ("a per-host asks/changelog document", _frag("docs/", "sha", "rd-")),
        ("the orchestrator pointer file", _frag("CLA", "UDE", ".md")),
        ("the agent-definitions directory (forward slash)", _frag(".", "cla", "ude/")),
        ("the agent-definitions directory (backslash)", _frag(".", "cla", "ude\\")),
        ("an upstream hand-over document", _frag("docs/", "upstream")),
        ("a second host's project directory name", _frag("TEX_", "compositor")),
    ]


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


#: G5 (FIX-GATE, v0.47.0 Phase C, R4#4): the leak class the path/host checks above cannot
#: see is a bare WORD that names no path and no host but still tells a public reader a
#: local-only process artifact exists (this project's own word for the document a lane
#: writes back to the orchestrator, built from pieces in `_BAREWORD_HASHES` below and never
#: spelled contiguously here) or names one of this project's own configured machines (drawn
#: from the standing box-identity records). Neither is a path, so `_fragments()`'s substring
#: scan cannot catch either, and `scan_host_name()`'s tokenizer (`\w+`) would itself SPLIT a
#: hyphenated machine name into two ordinary words and either miss it or flag an innocuous
#: one. Hashed, exact-case, same technique as the host name above.
_BAREWORD_RE = re.compile(r"\w+(?:-\w+)*")   # hyphen-aware: a hyphenated compound is ONE token

_BAREWORD_HASHES = frozenset(
    hashlib.sha256(w.encode("utf-8")).hexdigest()
    for w in (_frag("hand", "-back"), _frag("xavi", "-pc"), _frag("xavi", "_pc"))
)


def scan_bare_words(text: str) -> list:
    """`[(lineno, token)]` for every EXACT-CASE word/hyphen-compound token whose sha256
    lands in `_BAREWORD_HASHES`. `_BAREWORD_RE` (unlike `_WORD_RE` above) treats a
    hyphen-joined compound as one token, which is what lets it match a hyphenated machine
    name, or the local-only-artifact word above, as a whole instead of two unrelated
    ordinary words."""
    found = []
    for n, line in enumerate(text.splitlines(), 1):
        for tok in _BAREWORD_RE.findall(line):
            if hashlib.sha256(tok.encode("utf-8")).hexdigest() in _BAREWORD_HASHES:
                found.append((n, tok))
    return found


#: The down-only budget this ratchet INHERITS: every tracked *.py file that already carried
#: the local-only-artifact word above at the moment this row landed (v0.47.0 Phase C,
#: FIX-GATE), named explicitly rather than swept behind a blanket allowlist. A file's number
#: may only move DOWN from here (FIX-PACE's own rewrite of `pacing.py`'s ten occurrences is
#: exactly that kind of move) -- raising one, or a file absent from this table carrying any
#: hit at all, is what reds. No machine name has a nonzero budget: none was found in a
#: tracked *.py file when this landed, so any future one is a leak from day one, not a debt
#: to inherit.
_BAREWORD_BUDGET = {
    "tex_runtime/pacing.py": 10,
    "tools/gate.py": 1,
    "tex_runtime/stdlib_core.py": 1,
    "tex_runtime/graphed.py": 1,
    "tests/test_v044_cancel44.py": 1,
    "tests/test_v043_rider_a_capture_pending.py": 1,
    "tests/test_v040_phase1.py": 1,
    "tests/test_simp3_skip_budget.py": 1,
    "tests/test_simp3_consumer_registries.py": 1,
    "tests/test_seam45_embedding_host_seam.py": 1,
    "tests/test_pace45_pacing.py": 1,
    "tests/test_bench2_counts.py": 1,
    "benchmarks/preempt_drain_bench.py": 1,
    "benchmarks/host_path_counts.py": 1,
}


def test_lint1_g5_no_new_bare_word_leak(r: SubTestResult):
    print("\n--- LINT-1 (G5): no tracked *.py file leaks a hashed bare word beyond its "
          "inherited down-only budget ---")
    paths = tracked_paths()
    if paths is None:
        r.skip("LINT-1 G5 bare-word budget",
               "this tree is not a git checkout, so the tracked set cannot be enumerated")
        return
    overs, seen = [], 0
    for rel in paths:
        if not rel.endswith(".py"):
            continue
        p = _PKG / rel
        if not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        hits = scan_bare_words(text)
        if not hits:
            continue
        seen += 1
        budget = _BAREWORD_BUDGET.get(rel, 0)
        if len(hits) > budget:
            lines = [n for n, _ in hits]
            overs.append(f"{rel}: {len(hits)} occurrence(s) (budget {budget}, lines {lines})")
    if overs:
        r.fail("LINT-1 G5 bare-word budget",
               f"{len(overs)} tracked .py file(s) exceed their down-only bare-word budget "
               f"-- either a new leak, or a budget that must be lowered on purpose instead:\n  "
               + "\n  ".join(overs))
        return
    r.ok(f"every tracked .py file naming a budgeted bare word stays at or under its budget "
         f"({seen} file(s) carry one)")


def test_lint1_g5_bare_word_lint_is_not_inert(r: SubTestResult):
    print("\n--- LINT-1 (G5): the bare-word patterns fire on the real shape, and only on it ---")
    must_red = [
        "See the " + _frag("hand", "-back") + " for details.",
        "measured overnight on " + _frag("xavi", "-pc") + ".",
        "the key file " + _frag("xavi", "_pc") + " lives under ~/.ssh.",
    ]
    must_stay_green = [
        "the courier will hand back the package tomorrow.",  # two ordinary words, no hyphen
        "back to the drawing board.",                          # bare "back" alone
        "a hand truck moves the crate.",                        # bare "hand" alone
        _frag("xavi", "zzz") + " is one token.",               # the fragment is a substring, not a token
        "the pc tower needs a new fan.",                        # bare "pc" alone
    ]
    missed = [w for w in must_red if not scan_bare_words(w)]
    tripped = [f"{w}  ->  {scan_bare_words(w)[0][1]!r}"
               for w in must_stay_green if scan_bare_words(w)]
    if missed:
        r.fail("LINT-1 G5 bare-word witness (inert)",
               "the bare-word patterns did not fire on a real shape:\n  " + "\n  ".join(missed))
    elif tripped:
        r.fail("LINT-1 G5 bare-word witness (over-tight)",
               "the bare-word patterns fired on an unrelated neighbour:\n  " + "\n  ".join(tripped))
    else:
        r.ok(f"{len(must_red)} bare-word shapes red, {len(must_stay_green)} neighbours stay green")


#: H1 (FIX-HYGIENE, v0.50.0 Phase C, B4#1): a host tracker id must never sit in any tracked
#: file, in EITHER spelling this tree has actually leaked: a hyphen right after the
#: two-letter prefix with no arm suffix, or no hyphen after the prefix but a "-T<digits>"
#: arm suffix glued straight onto the digits. A prior gate (AUTOSAFE-50's own #8) grepped
#: only the hyphenated shape, so the hyphen-less spelling sailed straight past it -- the
#: bug this ratchet exists to close. A real regex, not a hash set, because the id space is
#: open (any digit run of 2-4 digits, matching every id this tree has ever used); the
#: digit-length floor of 2 keeps an unrelated two-letter+single-digit token (a chemical
#: formula, say) from ever matching. The two spellings are only ever assembled from pieces
#: at call time (see the witness test below), never written whole in this file's own
#: source, so this ratchet cannot fire on itself.
_TRACKER_PREFIX = _frag("C", "O")
_TRACKER_ID_RE = re.compile(r"\b" + _TRACKER_PREFIX + r"-?\d{2,4}(?:-T\d{1,3})?\b")

#: Down-only budget: zero, unconditionally. Unlike G5's per-file table above, there is no
#: inherited allowance to preserve -- every occurrence found at the moment this landed is a
#: leak to remove, not a debt to grandfather. The one exception is a single tracked file
#: this fix does not itself edit (a concurrent fix removes its one id in the same release);
#: it is named here, not budgeted, so the reason travels with the code instead of a bare
#: number.
_TRACKER_ID_ALLOWLIST = {
    # tex_runtime/compiled.py: removed there in the same release by the fix that owns
    # that module. Allowed here, once, by name, so this ratchet does not red on a removal
    # already in flight elsewhere in the same v0.50.0 Phase C batch.
    "tex_runtime/compiled.py",
}


def scan_tracker_ids(text: str) -> list:
    """`[(lineno, token)]` for every host tracker id, either spelling, found by regex --
    unlike `scan_host_name`/`scan_bare_words` above, the id space is open (any digit run),
    so a fixed hash set cannot enumerate it."""
    found = []
    for n, line in enumerate(text.splitlines(), 1):
        for m in _TRACKER_ID_RE.finditer(line):
            found.append((n, m.group(0)))
    return found


def test_lint1_h1_no_tracked_file_names_a_host_tracker_id(r: SubTestResult):
    print("\n--- LINT-1 (H1): no tracked file names a host tracker id, either spelling "
          "(down-only budget: zero) ---")
    paths = tracked_paths()
    if paths is None:
        r.skip("LINT-1 H1 tracker-id lint",
               "this tree is not a git checkout, so the tracked set cannot be enumerated")
        return
    hits, scanned = [], 0
    for rel in paths:
        if rel in _TRACKER_ID_ALLOWLIST:
            continue
        p = _PKG / rel
        if not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        scanned += 1
        found = scan_tracker_ids(text)
        hits += [f"{rel}:{n}: names a host tracker id — {tok!r}" for n, tok in found]
    if hits:
        r.fail("LINT-1 H1 tracker-id lint",
               f"{len(hits)} tracked line(s) name a host tracker id; the budget is zero:\n  "
               + "\n  ".join(hits[:40]))
        return
    r.ok(f"{scanned} tracked text file(s) name no host tracker id "
         f"(allowlist: {len(_TRACKER_ID_ALLOWLIST)})")


def test_lint1_h1_tracker_id_lint_is_not_inert(r: SubTestResult):
    print("\n--- LINT-1 (H1): the tracker-id pattern fires on both spellings, and only on "
          "them ---")
    must_red = [
        "tracked under " + _frag("C", "O") + "-187 upstream.",           # hyphen, no arm
        "confirmed against " + _frag("C", "O") + "187-T3 directly.",     # no hyphen, with arm
        "closed as " + _frag("C", "O") + "-42 last week.",               # hyphen, 2 digits
    ]
    must_stay_green = [
        "the company picked up the contract.",        # "co" inside a longer word
        "a coordinate frame, not a ticket.",           # "co" inside a longer word
        _frag("C", "O") + "2 emissions were measured.",  # single digit, below the floor
        "the budget item " + _frag("C", "O") + "-### is a placeholder.",  # literal hashes, no digits
        "the co-op meets Tuesdays.",                   # hyphen but no digits at all
    ]
    missed = [w for w in must_red if not scan_tracker_ids(w)]
    tripped = [f"{w}  ->  {scan_tracker_ids(w)[0][1]!r}"
               for w in must_stay_green if scan_tracker_ids(w)]
    if missed:
        r.fail("LINT-1 H1 tracker-id witness (inert)",
               "the tracker-id pattern did not fire on a real leaked shape:\n  "
               + "\n  ".join(missed))
    elif tripped:
        r.fail("LINT-1 H1 tracker-id witness (over-tight)",
               "the tracker-id pattern fired on an unrelated neighbour:\n  "
               + "\n  ".join(tripped))
    else:
        r.ok(f"{len(must_red)} tracker-id shapes (both spellings) red, "
             f"{len(must_stay_green)} neighbours stay green")


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
        _frag("docs/", "sha", "rd-") + "asks.md tracks status by ask id.",
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
        "sha" + "rd-asks is a familiar SUFFIX, not this fragment.",  # no leading "docs/"
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
