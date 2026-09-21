# Brief conventions — how to hand a lane a map it can trust

A "brief" here is the written ask one contributor hands another — human or agent — to
implement one change on one branch: what to build, where the code is, which gates to run.
This document is the form that ask takes.

It exists because of a measurement, not a preference. Across thirteen consecutive lanes of
one release round, **every lane landed** — and **fourteen separate incidents** were caused
by something the brief said that was not true at head:

| class | count | what it looked like |
|---|---:|---|
| wrong file | 2 | a bug attributed to a file that does not contain the code |
| wrong function | 2 | a real line number inside the wrong symbol |
| wrong count / incomplete enumeration | 3 | "four call sites"; there were five |
| false premise | 3 | a regression that did not exist; a soundness claim that was false as written |
| unachievable or over-tight gate | 2 | "0 rows may move" for a change that moves rows by construction |
| line-number drift | 1 | a citation landing on a blank line — repaired twice, rotted twice |
| brief requires an edit the implementer may not make | 1 | three lanes owed text in a file they are forbidden to edit |

None of these is a coding mistake. Every one is a **claim in a document that no test could
have caught**, and the cheapest of them cost an hour proving a bug did not exist. The nine
conventions below each name the incidents they prevent. `tools/check_citations.py` now
machine-checks the first of them across the shipped documents.

---

## The nine conventions

### 1. Name symbols, never lines

```text
    write this:   tex_roi.frame_window
    not this:     tex_roi.py:660
```

A symbol name survives every edit that moves the line; a line number is wrong the moment
anyone inserts a paragraph above it. (The pair above sits in a fenced block on purpose: a
citation inside a fence is output, not a claim, so the checker below skips it — which is
also how a pasted traceback stays out of the count.)
Where a line genuinely is the anchor — a specific statement inside a long function —
quote the line's **text** as well, the way the mutation harness does, so a reader can find
it by searching when it moves.

A `file.py:NNN` citation in a document is still allowed, and often useful, on one
condition: **the sentence that cites it names the symbol the line is inside**. That rule is
checked by `tools/check_citations.py` and gated by `tests/test_simp5_citations.py`; a line
that does not exist, or that is blank, is an error, and a citation whose sentence names no
covering symbol counts against a budget that only moves down.

*Prevents:* wrong function at a real line; line-number drift.

### 2. State the base sha, and the `wc -l` of every file the lane will touch

The brief gives the base sha once, at the top, and a size for every file it points at. The
hand-back reports the head figure beside it. This is already a de-facto habit — every
size-budget section in a hand-back does it — so make it a field rather than a courtesy.
Two things fall out for free: the implementer notices immediately when the file has moved
under the brief, and a module approaching its size budget is visible before the lane plans
against a number that no longer has room.

*Prevents:* line-number drift; a lane planning against stale headroom.

### 3. The implementer verifies every pointer BEFORE acting, and reports deviations

Not after, and not only when something looks wrong. Every `file`, `symbol` and `count` the
brief states is checked against the tree at the base sha as the lane's first act, and the
result goes in a fixed `## Pointer deviations` section of the hand-back: each pointer the
brief gave → **confirmed** / **corrected to X** → one line of evidence.

Where this was done voluntarily it found a wrong pointer **both times**. Where it was not,
a lane spent its first hour proving a bug did not exist.

*Prevents:* wrong file; wrong function; wrong count; a false plan premise carried into code.

### 4. A count in a brief is a claim, not a fact

Any "N call sites", "N consumers", "N rows" in a brief arrives **with the command that
derives it**, and the implementer re-runs that command and reports the number it returns.
If the numbers differ, the difference is the finding — not a rounding error to absorb.

One lane costed a fix at four call sites; there were five, and keying only the named two
would have left the public surface stale. Another named one anchor to re-point; a
three-line sweep found a second.

*Prevents:* wrong count; incomplete enumeration.

### 5. A gate must be achievable on a green tree

The brief states the gate **and the expected reading of that gate at the base sha**. If the
base reading is not zero/clean, the brief says what is allowed to move and why.

"Zero rows may move" is unachievable for a change that adds a call or moves a module, and
a gate that cannot be met is read past by hand — which is how a gate becomes decoration. An
over-tight gate is the same failure wearing a nicer suit: the reading has to be *argued*
rather than *read*, and the argument is what nobody will repeat next time.

*Prevents:* unachievable gates; over-tight gates.

### 6. The known-red list is a file reference, not prose

A brief cites the repository's known-red list; it does not retype it. A list living in
prose goes stale silently, and the failure mode is the worst one available: briefs named
two known reds when only one fired, for seven lanes running, which trains the reader to
skim the red list and wave it through. A second red that nobody recorded then arrives
looking exactly like the first.

*Prevents:* a stale known-red list; "green" meaning "I agreed with the error list".

### 7. A claim that opens a lane carries its null control

If a brief asserts a regression, it quotes the null measurement beside it — the same
comparison run against an unchanged tree. One lane was opened on a reported slowdown whose
own reference sitting read 1.082 on the null and 0.999 on the base leg: the entire lane
existed because a number was read without its control, and no defect was ever found.

*Prevents:* a false plan premise; a lane spent on a defect that does not exist.

### 8. A soundness claim inherited from an earlier finding is re-checked, never carried

"Finding F3 proved X, so we can assume X" is how a false claim propagates between lanes
with increasing confidence and no new evidence. A brief that depends on an earlier
conclusion restates it as a **premise you must verify**, with the check that settles it.

Every brief therefore carries a short `Premises you must verify` list: each load-bearing
assumption, and the command or read that confirms it. An assumption not on that list is
one the lane is not allowed to depend on.

Two incidents came from exactly this: a "provable core" that was false as written because
the handler it relied on swallowed every exception and returned the conservative answer,
and a plan step asserting a function could move alone when it reads a module global.

*Prevents:* a false soundness premise carried forward; a plan step that cannot be executed.

### 9. A `## Law edits owed` section, empty when there are none

Some files are off-limits to an implementer by standing rule — the invariants file, the
changelog, the tracker. A brief that requires text in one of them cannot be finished by the
lane that discovers the need, and the staleness that results is invisible: the drift checks
pass without the edit.

So every hand-back carries a `## Law edits owed` section listing the exact replacement text
for each such file, ready to paste, with an empty section being a valid and common answer.
Three lanes in one round each produced that text voluntarily; two of the three edits were
still unlanded a round later, because nothing collected them.

*Prevents:* a brief that requires an edit the implementer may not make.

---

## Three measurement rules that are not negotiable

These are not conventions — they are how a number on this box is allowed to be produced at
all. A brief that asks for a measurement states all three.

**Discard the first leg.** A comparison whose two legs share a compiled-artifact cache
directory is not a comparison: the second leg reads what the first one wrote. Two separate
lanes lost a measurement to exactly this, each producing a handful of rows that looked like
a structural change and were a warm cache. Give each leg its own cache directory, record
which directory each leg used and whether it started empty, and treat the first leg of any
sitting as a warm-up to be discarded rather than reported.

**Measure in a worktree, never in a host's plugin directory.** A checkout that lives inside
an embedding host's extension folder is loaded by that host, which means another process
may be importing the very tree being measured. Lanes run in their own worktree, outside any
directory a host scans, and measurements are taken from that worktree's parent so the
package resolves to the lane's copy and not to the installed one. A timing taken from the
installed tree is not a timing of the branch.

**Name the box beside the figure, every time the figure is republished.** Not once in a
standing caveat — beside the number, wherever the number is quoted. This rule was bought by
getting it wrong. A hand-back to a vendoring host asserted that *every* published TEX timing
figure came from the development laptop; that host read it against TEX's own documents, which
label the figures it had cited as an older Turing box, and said so. Both could not be true. The
document was wrong and the labels were right: this engine was calibrated on one card, most
published figures are that card's, several roadmap *targets* name the later laptop, and the
v0.38.0 sitting is the older box again. Nobody was misled, because the per-figure labels held —
which is the whole argument for the rule. A sweeping claim about a set of figures is a claim
about every member of the set, and it will be checked by the one reader who has the set in
front of them. Quote the number, name its box, or do not republish it.

---

## The hand-back skeleton

Copy this whole block. Every section stays, including the ones whose honest content is
"none" — an empty section is an answer, a missing section is a silence.

```markdown
# <ASK-ID> — hand-back

Branch: <branch>
Base sha: <sha>          Head sha: <sha>
Worktree: <path>

## What landed
One paragraph, then the row the release record will copy:
<ask id> → landed / declined (reason) → <file>:<symbol> → <the check named in the ask>

## What did NOT ship
Stated as plainly as what did — declined items, deferred halves, anything the ask asked
for that is not in the diff, each with its reason.

## Pointer deviations
| pointer the brief gave | verdict | evidence |
|---|---|---|
| `mod.symbol` | confirmed | <one line> |
| `other.thing` | corrected to `other.actual` | <one line> |
(Empty table = every pointer verified and correct. "Not checked" is not a verdict.)

## Premises verified
Each load-bearing assumption the brief stated, and what settled it.

## Sizes
| file | wc -l at base | wc -l at head | budget / floor |
|---|---:|---:|---|

## Gates
| # | gate | rc | summary line (verbatim) |
|---|---|---:|---|
Cheapest first. Real exit codes, captured on the line after the run and never after a pipe.
The expected base reading for each gate comes from the brief; say where the head reading
differs from it.

## Law edits owed
Exact replacement text for any file this lane may not edit, ready to paste, with the file
and the section named. Empty is a valid answer — write "none".

## Findings outside this ask
Filed separately; listed here by id and one line each.
```

## What a release note owes a vendoring host

A second host vendors a pinned subset of this tree and adopts upstream work only by re-pinning
whole commits, so it reads release notes the way a compiler reads a header. Asked what maximum
drift it wanted, it declined to name one: not a commit count and not a minor count, because the
number is not what makes a re-pin plannable. Four things are, and each is named **before the
tag**, in the release entry rather than only in the diff:

1. **Any newly reserved name.** Defining a function of a reserved name fails `E3011`, so
   reserving one can break source a host already ships. A host cannot scan for a name it has
   not been told about.
2. **Any grammar-visible change, and any move of `LANGUAGE_VERSION`.** A host's on-disk tool
   manifests carry the language version as a literal, and a host may hash that literal into a
   tool's trust digest — in which case moving it re-signs every tool that host ships, and the
   failure direction is a demoted tool rather than a stale grant.
3. **Any default that moves**, engine-side or host-facing.
4. **Any new module filename.** A vendoring step is typically a directory copy, so a new file
   arrives in a host's tree whether or not that host imports it.

Two rules of form travel with them. State the delta **from the host's actual pin** as well as
from the previous release: a host several minors behind cannot compose four changelogs into an
answer, and the composition is exactly where a reserved name gets lost. And mark anything that
exists only at head as **head-only since vX.Y**, so a host can plan against it without mistaking
it for something it can call at its pin.

## The brief skeleton, in one paragraph

A brief states: the ask id and the one change; the branch, the worktree path and the base
sha; the files it points at **by symbol**, each with its `wc -l` at that sha; every count it
claims **with the command that derives it**; the gates in cheapest-first order **with their
expected reading at the base sha**; the `Premises you must verify` list; which files or
sections other concurrent lanes own; and where the hand-back goes. It does not restate
standing law — it points at it. A brief that cannot say which premises are load-bearing is
not yet a brief.
