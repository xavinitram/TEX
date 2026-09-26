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
#: Re-pinned from 105 to 106 (TRK-154): `test_trk154_call_site_capture_decisions.py::
#: test_trk154_the_flip_was_never_a_working_capture` drives a REAL `run_graphed` capture
#: attempt on a program whose pre-fix static verdict was "capturable" — the same
#: no-CPU-witness reason as the v042-graph rows above, since there is no CPU stand-in for
#: a genuine `cudaErrorStreamCaptureInvalidated`.
#: Re-pinned from 106 to 107 (v041-p2, TRK-67): `test_perf2_host_scalar.py`'s
#: `test_trk67_string_family_costs_no_readback` counts real `torch.Tensor.item()` calls
#: split by device — the same no-CPU-witness reason
#: `test_perf2_a_host_scalar_costs_no_readback` (already inside the pin) carries: a CPU
#: `.item()` is a host-memory read, not a device round trip, so the row would pass
#: without measuring anything off CUDA.
#: Re-pinned from 107 to 108 (v041-p2, TRK-68): the same file's
#: `test_trk68_array_index_and_loop_bound_cost_no_readback` carries the identical
#: no-CPU-witness reason for a `$param` array index / loop bound's device-readback count.
#: Re-pinned from 108 to 104 (v042-noviewer): `viewer_exposure()`/`viewer_gamma()` and the
#: whole `viewer_context=`/CUDA-graph host-context-buffer mechanism are removed outright —
#: the ComfyUI node never exposed a viewer input and the embedding host grades in its own
#: shader, so nothing ever consumed them (the author's ruling). Their two test files are
#: deleted rather than repaired: `tests/test_v040_phase2.py` (PM-11, 1 site —
#: `test_pm11_oom_rung_path`) and `tests/test_v042_graph.py` (v042-graph, the 3 sites the
#: 102→105 re-pin above named) — 4 sites gone. The replacement, `tests/test_v042_noviewer.py`
#: (proving the names are no longer reserved, the kwarg is rejected everywhere, and a
#: non-viewer program's lineage-key byte format is unmoved), needs no CUDA witness — every
#: row is AST-level or a pure-function check — so it adds 0. Net 108 − 4 = 104.
#: Re-pinned from 104 to 105 (v042-hostaudit): `test_v042_hostaudit4a_oom_refusal.py::
#: test_hostaudit4a_unrecoverable_oom_carries_a_refusal` skips when this torch build has no
#: `cuda.OutOfMemoryError` type — the same no-witness reason `test_eng2_oom_ladder`'s own
#: identical guard (already inside the pin) carries: without that type there is nothing to
#: monkeypatch `_run_tier` into raising, so the row cannot exercise the OOM ladder at all,
#: real device or not.
#: Re-pinned from 105 to 106 (v042-floor, TRK-84, rebased onto v042-hostaudit):
#: `test_trk84_coord_ramp_bitexact.py::test_trk84_pooled_instance_across_devices_never_mixes_ramps`
#: needs a real CUDA device to prove a pooled Interpreter's `_coord_ramps` cache never leaks
#: a CPU-built ramp into a later CUDA cook on the same instance — no CPU stand-in exists for
#: that cross-device reuse, so a CUDA-absent run reports a genuine skip. This lane's own
#: delta has always been +1 (108→109 originally); rebased twice (v042-noviewer's 108→104,
#: then v042-hostaudit's 104→105), the honest resolved value is 105+1 = 106.
#: Re-pinned from 106 to 107 (v0422-race, TRK-178): `test_v0422_race.py::
#: test_v0422_race_restore_pinned_h2d_survives_concurrent_readers` skips when there is no CUDA
#: device — the row stresses `ResultCache._restore`'s pinned non-blocking host-to-device copy,
#: which only exists on the CUDA leg; there is no CPU-side witness for a DMA-engine copy that
#: does not happen on CPU at all.
#: Re-pinned from 107 to 108 (v0422-race, TRK-178): the coordinator's follow-up hardening added
#: `test_v0422_race.py::test_v0422_race_restore_pinned_h2d_fences_a_foreign_stream`, which skips
#: for the same CUDA-only reason as the row above — a foreign-CUDA-stream fence has no meaning
#: without a CUDA device to fence on.
#: Re-pinned from 108 to 111 (v0.43.0, TOOL-7 cleanup): five new rows landed with this release;
#: two were precondition guards wearing a skip they never needed.
#: `test_v043_tool7_warm.py`'s "stock 'grade' no longer has 2 warm variants" and "fresh image
#: tool no longer has 2+ warm keys" rows are facts about a fixed stock exemplar and a freshly-
#: built fixture, not about this box — if either ever stopped holding, that is a real
#: regression that belongs in front of the reader, not behind a skip. Both are now `r.fail`,
#: the witness shape this ratchet asks for, and neither counts here any more. The remaining
#: three genuinely need an absent environment: `test_v043_tool7_warm.py::
#: test_tool7_warm_status_capturable_field` and `test_v043_rider_b_ingest_merge.py`'s ingest
#: row both need a real CUDA device — no CPU witness exists for a graph-capturability verdict
#: or a pinned H2D leg, the same reason every other CUDA-only row already in this pin carries.
#: `test_lint1_no_local_only_path_refs.py` needs a working git checkout: it enumerates the
#: PUSHED set via `git ls-files` through `tracked_paths()` (the same helper
#: `test_simp3_no_machine_paths.py` uses and skips the same way when it is absent). A
#: filesystem-walk fallback was considered and rejected, not merely skipped over: without git
#: there is no way to tell a TRACKED path from a local-only one — exactly the distinction this
#: row exists to police — so walking the working tree instead would read the project's own
#: excluded orchestration files, including prose that legitimately names the very fragments
#: this row forbids (this hand-back among them), and manufacture the false positives the row
#: exists to prevent rather than remove its need for git. 108 + 3 = 111.
#: Re-pinned from 111 to 113 (PACE-45(b)): two CUDA-only rows in `test_pace45_pacing.py`, the
#: paced-cancellation repro's non-timing and timing halves
#: (`test_pace45_cuda_pacing_bit_exact_and_repro`, `test_pace45_cuda_repro_latency`) — no CPU
#: witness exists for "the host queued ahead of the device", since a CPU cook has no such
#: async queue to get ahead of at all. Every other row in that file (the opt-in gate's own
#: unit test, the CPU bit-exactness check) needs no CUDA and carries no `r.skip`.
#: Re-pinned from 113 to 114 (PACE-45(a)): `test_pace45_done_event.py::test_pace45_done_event_cuda`
#: needs a real CUDA device too — there is no CPU `torch.cuda.Event` to witness. Its CPU
#: sibling (`test_pace45_done_event_cpu`) needs none and carries no `r.skip`.
#: Re-pinned from 114 to 116 (FUSEDDEV-46): `test_fuseddev46_device.py` adds two rows that
#: each need a real CUDA device to have a SECOND device to mismatch against at all — a
#: CPU-only cook has nothing to disagree with, so there is no CPU witness for
#: "cuda:0 and cpu" (`test_fuseddev46_fused_torch_compile_cuda_stays_on_device`, the fused
#: chain, and `test_fuseddev46_single_node_torch_compile_cuda_stays_on_device`, the same
#: crash on a plain unfused node). The file's third row, the non-skip twin over
#: `helpers.devices()`, needs no `r.skip` at all — it is the "loop, not a skip" idiom this
#: budget's own header names, and runs the same-device case everywhere and the real
#: cross-device case wherever CUDA happens to be present.
#: Re-pinned from 116 to 117 (v046-c-gate, FIX-GATE/G2): `test_v046_fixgate.py::
#: test_g2_enumerate_paths_sees_an_untracked_not_ignored_file` needs a working git checkout
#: (to write a probe file and confirm `git status --porcelain` itself reports it untracked
#: before trusting the enumeration under test) — the exact same dependency, and the same
#: reasoning, as `test_lint1_no_local_only_path_refs.py`'s row already inside this pin: there
#: is no meaningful witness for "an untracked-not-ignored file is enumerated" without git to
#: make a file actually untracked-not-ignored in the first place.
#: Re-pinned from 117 to 118 (PACE-462): `test_pace462_bounded_lookahead.py::
#: test_pace462_cuda_drained_bound` needs a real CUDA device — the same no-CPU-witness reason
#: PACE-45(b)'s CUDA-only rows already carry: there is no CPU async queue for the host to get
#: ahead of, so "how much device work is left behind after a pre-emption" has nothing to
#: measure off CUDA. Every other row in that file drives the ring/depth mechanism through a
#: mocked `torch.cuda` (the same `_DeviceSpy` shape `test_fixobsroute46_pacing.py` uses) and
#: needs no `r.skip`.
#: Re-pinned from 118 to 119 (PROF-462): `test_prof462_device_honest.py::
#: test_prof462_real_cuda_cook_resolves_device_ms` needs a real CUDA device — it is the one
#: row in that file that is NOT a fake-`torch.cuda.Event` mechanism test (those need no
#: device at all and carry no `r.skip`), and there is no CPU witness for "a sampled cook's
#: device time resolves via a real CUDA event", the same reason every other CUDA-only row in
#: this pin carries.
_SKIP_BUDGET = 119


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
