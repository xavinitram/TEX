"""An absent environment is a SKIP, and the number of them is a ratchet.

The suite's only automated lane is CPU-only, and 46 of its files carry a `cuda.is_available`
gate. For years the gate's absent side answered `r.ok("… (no GPU, SKIPPED)")` — so the lane's
`N passed` counted, as passes, exactly the rows it could not run. Two of those rows are
invariant 9's named enforcers, which made the invariant enforced nowhere automatically while
every reading said green. A pass that measured nothing is worse than a missing test: it is a
missing test that reports a guarantee.

`SubTestResult.skip(name, reason)` has existed the whole time and says the true thing. These
two rows make it the only thing that can be said.

**Arm (a) — no absent environment is reported as a pass.** An AST census flags any block whose
ONLY action is an `r.ok(...)` whose message uses the skip vocabulary (`SKIPPED`, `skipped (…)`,
`(no CUDA/GPU …)`, `not measured`, `absent in this environment`). The pin is ZERO. It is
deliberately a VOCABULARY check rather than an attempt to infer intent from the guard: a guard
like `if _CUDA:` appears on both sides of the line — `test_v028_phase1`'s CPU arm really does
run its benchmark and really does pass — and a checker that guessed would either miss the
honest half or red on it. What the vocabulary catches is the shape that actually existed: a
site that TELLS the reader it skipped while telling the counter it passed. Say it in the
message and you must say it with `r.skip`.

**Arm (b) — the skip budget.** Every `r.skip(name, reason)` site in `tests/` is counted and the
total is pinned. Each one is a row the CI lane cannot run, so the pin is a debt figure and the
ratchet runs the PUB-1 way: over the pin reds (a new unrunnable row is a decision, not a
reflex), under it reds too (a stale pin is the same lie as a missing one — re-pin DOWN and the
debt is recorded as repaid). Every site must also pass a non-empty reason, because a skip
without one is only a quieter way of saying nothing.

Why a source census and not a count of pytest's own skip lines from a CI-shaped run: a run's
skip count is a fact about the BOX, not about the tree — this box has CUDA and the CI lane does
not, so the same tree yields two different numbers and a pin would mean something different in
each lane. It would also put a whole-suite run (~3 min) inside the cheap tier, to measure
something the sources already state exactly. The census is milliseconds, imports no product
module, and answers the same in every environment.
"""
import ast
import pathlib
import re

from helpers import SubTestResult

_TESTS = pathlib.Path(__file__).resolve().parent

#: The vocabulary a site uses when it is telling the reader it did not run. Case-sensitive on
#: `SKIPPED` on purpose: "skipped, not merely dropped" is prose ABOUT a product behaviour
#: (`test_v0332_audit`'s H6 row), not a report about this run.
_SKIP_VOCAB = re.compile(
    r"SKIPPED"
    r"|\bskipped \("
    r"|\(skipped\b"
    r"|\(no (CUDA|GPU)\b"
    r"|\bno (CUDA|GPU) (on|to|-)"
    r"|\bnot measured\b"
    r"|\babsent in (this environment|test venv)\b")

#: Rows the CPU lane cannot run, counted at their sites. Moves DOWN freely (and reds until the
#: pin follows); moves UP only as a deliberate decision that says which environment the new row
#: needs and why the row cannot be written without it.
#: NEG-4 raised this by 2: `test_neg4_citation_root_through_a_link` needs a working git
#: binary (to build its throwaway repo) and a platform that can create a directory
#: junction or symlink (to reproduce the root-through-a-link condition at all) — neither
#: is guaranteed on every CI runner, and the row must report a real skip rather than a
#: silent pass when either is absent (see `tests/test_simp5_citations.py`).
#: Re-pinned from 97 to 101: `census_skips` already counted 100 sites on the unrepaired
#: tree (a 3-row gap the pin had not caught up to), and repairing arm (a) above — the
#: return-in-function-in-loop codegen-unsupported arm that used to report its skip through
#: `r.ok`'s vocabulary — turns it into a genuine 101st site: it was always a skip, just
#: never counted as one. None of the newly-counted sites has a witness that runs without the
#: missing capability (an oracle/codegen path for the case in question), so raising the pin
#: is the honest move rather than inventing one.
#: Re-pinned from 101 to 102 (PM-11): `test_v040_phase2.py::test_pm11_oom_rung_path` needs a
#: real CUDA device — `tex_engine._oom_retry`'s rung 2 is gated on
#: `str(ctx.device).startswith("cuda")` and cannot be exercised any other way — and the same
#: arm (a) repair applies here too: this row used to report the absence through `r.ok`'s
#: vocabulary (a pass that measured nothing), which is what surfaced it. CI's lane is CPU-only,
#: so this row skips there; no witness exists that proves the OOM ladder's tiled rung without a
#: CUDA device to be short of memory on, so raising the pin is the honest move, not writing one.
#: Re-pinned from 102 to 105 (v042-graph): `tests/test_v042_graph.py`'s three CUDA-graph
#: capture/replay rows (`test_v042_viewer_replay_correctness`,
#: `test_v042_viewer_no_recapture`, `test_v042_plain_program_unaffected`'s replay half) each
#: need a real CUDA device to capture and replay a graph at all — there is no CPU witness
#: for a `torch.cuda.CUDAGraph`, so an absent-CUDA run reports a genuine skip, not a
#: reflex. The fourth new row (`test_v042_viewer_now_capturable`) needs no device — it
#: drives the static AST gate (`graphed._capturable`) directly — and carries no `r.skip`.
_SKIP_BUDGET = 105


def _literal(node) -> str:
    """Every string literal inside an expression, joined — an f-string's fixed text included."""
    return " ".join(s.value for s in ast.walk(node)
                    if isinstance(s, ast.Constant) and isinstance(s.value, str))


def _lone_report(block, attr: str):
    """The single `<name>.<attr>(...)` call a block makes, when the block does nothing else.

    A bare string expression (a comment-as-docstring) and a trailing `return` do not count as
    doing something; anything else does, which is what keeps a real assertion that merely
    mentions a skip out of the census.
    """
    body = [s for s in block
            if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    calls = [s.value for s in body
             if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
             and isinstance(s.value.func, ast.Attribute) and s.value.func.attr == attr
             and isinstance(s.value.func.value, ast.Name)]
    other = [s for s in body
             if not isinstance(s, ast.Return)
             and not (isinstance(s, ast.Expr) and s.value in calls)]
    return calls[0] if len(calls) == 1 and not other else None


def _test_files():
    return sorted(p for p in _TESTS.glob("test_*.py"))


def census_ok_as_skip(files) -> list:
    """`[(file, lineno, message)]` for every block that reports a skip as a pass."""
    found = []
    for p in files:
        src = p.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.If):
                blocks = (node.body, node.orelse)
            elif isinstance(node, ast.ExceptHandler):
                blocks = (node.body,)
            else:
                continue
            for block in blocks:
                call = _lone_report(block, "ok")
                if call is None or not call.args:
                    continue
                msg = _literal(call.args[0])
                if _SKIP_VOCAB.search(msg):
                    found.append((p.name, call.lineno, msg))
    return found


def census_skips(files) -> tuple:
    """`(sites, reasonless)` — every `<name>.skip(...)` call, and those without a reason."""
    sites, reasonless = [], []
    for p in files:
        src = p.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(src)):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "skip" and isinstance(node.func.value, ast.Name)):
                continue
            sites.append((p.name, node.lineno))
            reason = None
            if len(node.args) >= 2:
                reason = node.args[1]
            else:
                for kw in node.keywords:
                    if kw.arg == "reason":
                        reason = kw.value
            if reason is None or not _literal(reason).strip():
                reasonless.append((p.name, node.lineno))
    return sites, reasonless


def test_simp3_an_absent_environment_is_never_reported_as_a_pass(r: SubTestResult):
    print("\n--- SIMP-3: no block reports a skip through r.ok (pin: 0) ---")
    found = census_ok_as_skip(_test_files())
    if found:
        r.fail("SIMP-3 skip-as-pass",
               f"{len(found)} site(s) in {len({f for f, _, _ in found})} file(s) say they "
               f"skipped and count as a pass — use r.skip(name, reason), which is neither a "
               f"pass nor a failure:\n  "
               + "\n  ".join(f"{f}:{n}  {m[:80]}" for f, n, m in found))
        return
    r.ok("no r.ok site in tests/ reports an absent environment")


def test_simp3_the_skip_budget_is_a_ratchet(r: SubTestResult):
    print(f"\n--- SIMP-3: r.skip sites carry a reason and stay at their pin ({_SKIP_BUDGET}) ---")
    sites, reasonless = census_skips(_test_files())
    if reasonless:
        r.fail("SIMP-3 skip reason",
               "r.skip(name, reason) without a reason is a quieter way of saying nothing:\n  "
               + "\n  ".join(f"{f}:{n}" for f, n in reasonless))
        return
    n = len(sites)
    if n > _SKIP_BUDGET:
        r.fail("SIMP-3 skip budget (new skip)",
               f"{n} skip sites, pinned {_SKIP_BUDGET} — a row the automated lane cannot run "
               f"is a deliberate decision: say which environment it needs and raise the pin, "
               f"or give the row a witness that runs without it")
    elif n < _SKIP_BUDGET:
        r.fail("SIMP-3 skip budget (stale pin)",
               f"{n} skip sites, pinned {_SKIP_BUDGET} — re-pin DOWN to {n}; a pin above the "
               f"truth reports debt that has already been repaid")
    else:
        r.ok(f"{n} r.skip sites in {len({f for f, _ in sites})} files, each with a reason, "
             f"at the pin")
