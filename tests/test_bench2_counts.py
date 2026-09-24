"""BENCH-2 — the per-tick structural counts of the interactive host paths, pinned.

`benchmarks/host_path_counts.py` measures how many times each seam on the cook path is
entered per interactive tick, driving TEX's own `examples/host_demo.py::RoiComp` (the ten-
stage comp, a host-armed CACHE-2 results cache, CACHE-1 lineage keys carrying the upstream
chain). This file turns those counts into a gate.

WHY COUNTS AND NOT TIMES. `docs/roadmap.md` §10 item 3 records the null controls measured on
the dev box: `eight_config_bench` run against a BYTE-IDENTICAL tree returns per-config geomeans
from 0.949 to 1.105, with individual rows spanning 0.70-2.32, and `cpu_off_warm` has tripped
the 0.95 stop-ship threshold against itself. No timing assertion can live in a suite that runs
on shared CI hardware. The counts below are exact integers that repeat tick after tick and
process after process — so they can.

WHAT MOVES A PIN. Every row carries a comment naming what legitimately changes it. A pin is
not a wish: when the engine genuinely does one more (or one fewer) of something per tick, the
pin is RE-DERIVED with the harness and the change is explained in the CHANGELOG. The failure
message says which row moved, from what to what, and prints the command that re-derives it.
The pin also fails on a DECREASE, for the REG-2 reason — a bound that only ever loosens is
decoration, and a count that silently drops usually means a spy stopped seeing its target.

SHAPE (roadmap §10.4): CANARY over the per-tick contract, with the cold-scenario MUTATION
GUARD (`test_bench2_counters_are_not_inert`) as its never-vacuous half — the ANIM-1 lesson,
where thirteen rows asserting "this counter is 0" all passed with the counters zeroed out.

PORTABILITY. CPU, no ComfyUI, no compiler, no CUDA. The CUDA row SKIPs (it does not pass)
without a device. Runs at 96^2 with a 48^2 window and four ticks to stay inside a few seconds.
"""
from helpers import *
# By name, not by star: `helpers.__all__` is pinned to its v0.35.0 set (HOOK-4), because
# `from helpers import *` is a surface a host's own suite binds.
from helpers import load_counts_harness


# The harness shape these pins were measured in. Changing any of these re-derives every pin:
# the window position walk depends on `TICKS` (see `Scenario._pan_roi`), and a resolution that
# crossed a tiling threshold would change the cook counts.
RES, WINDOW, TICKS = 96, 48, 4
REDERIVE = (f"python benchmarks/host_path_counts.py --device cpu --res {RES} "
            f"--window {WINDOW} --ticks {TICKS} --prof1 off")


def _bench():
    """The harness itself, loaded by path — see `helpers.load_counts_harness`.

    `benchmarks/` is `.comfyignore`d and is not a package, so there is no import name to use;
    a path load also keeps this test honest about measuring the harness the design note names
    rather than a copy of its logic. The loader moved to `helpers` when a SECOND copy of the
    harness's frame filter (in `tests/test_perf7_compiled_cold.py`) was found reading zero:
    one loader, one filter, no copies to drift."""
    return load_counts_harness()


def _api_counts(scenario_name: str, device: str = "cpu") -> dict:
    """Run ONE scenario's API pass and return `row -> (min, max)` per steady tick."""
    b = _bench()
    cls = next(c for c in b.SCENARIOS if c.name == scenario_name)
    scn = cls(RES, WINDOW, device, ticks=TICKS)
    try:
        folded = b.pass_api(scn, TICKS)
    finally:
        scn.teardown()
    return {row: (s["min"], s["max"]) for row, s in folded.items()}


def _check(r: SubTestResult, label: str, got: dict, pins: dict):
    bad = []
    for row, want in pins.items():
        lo, hi = got.get(row, (None, None))
        if lo is None:
            bad.append(f"{row}: MISSING from the harness's rows (a spy target was renamed "
                       f"or removed — the pin cannot be checked, which is a failure)")
        elif lo != hi:
            bad.append(f"{row}: UNSTABLE across the {TICKS} steady ticks ({lo}..{hi}); a row "
                       f"that disagrees with itself cannot gate — re-derive with: {REDERIVE}")
        elif lo != want:
            bad.append(f"{row}: {want} -> {lo} per tick")
    if bad:
        r.fail(f"BENCH-2 {label}",
               "; ".join(bad) + f" || re-derive with `{REDERIVE}` and explain the move in "
               f"CHANGELOG.md (a count that changed in either direction is a change to what "
               f"an embedding host pays per interactive tick)")
    else:
        r.ok(f"{label}: {len(pins)} per-tick rows hold at their pins")


# ── the pins ────────────────────────────────────────────────────────────────
# Measured with the command in REDERIVE at v0.37.0. Device-INDEPENDENT rows only: each of
# these reads the same on CPU and on CUDA (verified at 1024^2/512 on an sm_120 box), which is
# what makes them CI-gateable. Device-dependent counts (kernels, memcpys, allocations) are in
# `test_bench2_cuda_per_tick_counts` and skip without a device.

_TERMINAL = {
    # The terminal-knob scrub: viewport window open, `dirty_from` = the last stage, so nine
    # canvases stand and one stage cooks its window.
    "tex_engine.cook":         1,    # one dirty stage. Moves if the host's dirty-suffix walk
                                     # or `chain_windows`' decline logic changes.
    "TEXCache.compile_ast":    0,    # ANIM-1: a $param is a cook-time binding. A non-zero here
                                     # means a scrub recompiles — the bug ANIM-1 freezes out.
    "TEXCache.compile_tex":    1,    # one cached-compile lookup per cook.
    "Lexer.tokenize":          0,    # PERF-1 re-pin (was 1/1). `roi_plan -> _walk` still
    "Parser.parse":            0,    # misses its memo every tick — the walk's answer really
                                     # does depend on the param VALUES — but the front end
                                     # behind it now runs at most ONCE per source
                                     # (`tex_roi._pristine_program` + `ast_nodes.clone_tree`),
                                     # so a scrub re-folds a reused parse instead of
                                     # re-lexing one. A 1 here again means the parse memo
                                     # stopped being reached.
    "TypeChecker.check_collect": 0,  # lint only; a cook must never run the collecting checker.
    "tex_roi._fold_program":   1,    # STAYS 1, by design: the fold is what is value-dependent
                                     # (a `$sigma` in a halo radius, a `mix(@A,@B,$k)` arm
                                     # that `k = 0` erases), so it is the parse that was
                                     # memoized and not the fold. Only a proven
                                     # value-independent walk would take this to 0.
    "tex_roi.roi_plan":        2,    # 1 per engine cook + 1 for the host's own halo question
                                     # (`RoiComp._halo_of` misses its memo when params move).
    "tex_roi.chain_windows":   1,    # one plan per tick, whatever the dirty suffix.
    "tex_results.lineage_key": 11,   # 10 whole-frame chain keys + 1 windowed key for the
                                     # cooked stage. Host policy: minting the nine clean-prefix
                                     # keys every tick is what makes this 11 and not 2.
    "TEXCache.fingerprint":    1,    # PERF-5 re-pin (was 2). ONE per cook: `prepare` computes
                                     # the key for `_preflight_memory`'s memo and hands the
                                     # same string to the compile, which uses it for the cache
                                     # probe AND the store. A 2 here means a caller went back
                                     # to computing its own; a 0 means the spy stopped seeing a
                                     # `@staticmethod` (see the mutation guard below).
    "ResultCache.get":         1,    # one probe for the one dirty stage,
    "ResultCache.put":         1,    # one store. Growth per tick is host retention policy.
    "tex_memory.run_roi":      1,    # the cook took the ROI path. 0 would mean a whole-frame
                                     # fallback — a correctness-shaped regression, not a perf one.
    "Interpreter._exec_stmt":  2,    # the vignette stage's two statements, on the CPU
                                     # interpreter tier. Device-dependent ONLY in the sense
                                     # that a compiled tier would bypass it; on CPU at this
                                     # size the interpreter is the tier.
}

_MIDGRAPH = {
    # The same drag five nodes up: the dirty suffix is five stages, so every per-cook fixed
    # cost is paid five times. The difference from _TERMINAL is what one extra stage costs.
    "tex_engine.cook":         5,    # stages 5..9.
    "TEXCache.compile_ast":    0,    # ANIM-1 again, over five programs.
    "TEXCache.compile_tex":    5,
    "Lexer.tokenize":          0,    # PERF-1 re-pin (was 1/1 — ONE, not five, because only
    "Parser.parse":            0,    # the scrubbed stage's params moved). Now zero for the
                                     # same reason as `terminal`: five sources, five parses,
                                     # all of them already done.
    "tex_roi._fold_program":   1,    # still one fold, for the one stage whose params moved.
    "tex_roi.roi_plan":        6,    # 5 engine plans + 1 host halo question.
    "tex_roi.chain_windows":   1,
    "tex_results.lineage_key": 15,   # 10 chain keys + 5 windowed keys.
    "TEXCache.fingerprint":    5,    # PERF-5 re-pin (was 10): 1 per cook, five cooks.
    "ResultCache.get":         5,
    "ResultCache.put":         5,
    "tex_memory.run_roi":      5,    # all five stages stayed on the ROI path.
    "Interpreter._exec_stmt":  6,
}

_PAN = {
    # The window MOVES and no parameter changes. Every memo a scrub misses is a hit here, so
    # what is left is what moving the window itself costs.
    "tex_engine.cook":         1,
    "TEXCache.compile_ast":    0,
    "TEXCache.compile_tex":    1,
    "Lexer.tokenize":          0,    # params constant => the WALK memo hits, so not even the
    "Parser.parse":            0,    # fold runs. This scenario was the control that showed
    "tex_roi._fold_program":   0,    # the terminal scrub's pre-PERF-1 1/1 to be the memo key
                                     # and not the cook; it still separates "the walk memo
                                     # hit" (here) from "the parse memo hit" (terminal).
    "tex_roi.roi_plan":        1,    # the engine's, only: the host's halo memo hits.
    "tex_roi.chain_windows":   1,
    "tex_results.lineage_key": 11,
    "TEXCache.fingerprint":    1,    # PERF-5 re-pin (was 2): 1 per cook.
    "ResultCache.get":         1,
    "ResultCache.put":         1,    # 1, not 0: the window is in the key, so a moved window
                                     # is always a miss. A 0 would mean the walk revisited a
                                     # position and the scenario stopped being a pan.
    "tex_memory.run_roi":      1,
    "Interpreter._exec_stmt":  2,
}

_ALL_DIRTY = {
    # A knob on the FIRST stage with no window: every lineage key changes, so the results
    # cache misses all ten and the whole frame recooks.
    "tex_engine.cook":        10,
    "TEXCache.compile_ast":    0,    # ten cooks, zero compiles — ANIM-1 across the chain.
    "TEXCache.compile_tex":   10,
    "Lexer.tokenize":          0,    # no window => no ROI analysis at all,
    "Parser.parse":            0,
    "tex_roi._fold_program":   0,
    "tex_roi.roi_plan":        0,    # which is why a whole-frame recook never re-parses.
    "tex_roi.chain_windows":   0,
    "tex_results.lineage_key":10,    # one whole-frame key per stage; no windowed keys.
    "TEXCache.fingerprint":   10,    # PERF-5 re-pin (was 20): 1 per cook, ten cooks —
                                     # the clearest reading of the change, and the one
                                     # `docs/host-path-counts.md` §6 item 6 named.
    "ResultCache.get":        10,
    "ResultCache.put":        10,
    "tex_memory.run_roi":      0,    # no roi => the whole-frame path, by construction.
    "Interpreter._exec_stmt": 12,    # 12 statements across the ten stage programs.
}

_LINT = {
    # `tex_api.check` on a one-character edit — the editor's live-lint path, on the UI thread
    # between keystrokes. Everything cook-shaped must be 0 here.
    "tex_engine.cook":         0,
    "TEXCache.compile_ast":    0,    # LANG-2: check() is compile-ONLY diagnostics.
    "TEXCache.compile_tex":    0,
    "Lexer.tokenize":          1,    # exactly one lex + one parse per keystroke,
    "Parser.parse":            1,
    "TypeChecker.check_collect": 1,  # and one collecting type-check (check() is total).
    "TypeChecker.check":       0,    # NOT the raising `check()` — a lint must not raise.
    "tex_roi._fold_program":   0,
    "tex_roi.roi_plan":        0,
    "tex_results.lineage_key": 0,
    "TEXCache.fingerprint":    0,    # a lint never touches the program cache.
    "ResultCache.get":         0,
    "ResultCache.put":         0,
    "tex_memory.run_roi":      0,
    "Interpreter._exec_stmt":  0,
}


_NODE_SCRUB = {
    # BENCH-3, from PERF-4's finding F5. The seven scenarios above drive `tex_api` /
    # `tex_engine` directly, where `forgive_dead_refs` is False — and that flag is the ONLY
    # door to the lazy tier, so a regression on the ComfyUI node's own per-tick cost could not
    # move any row above. This scenario drives what a user's slider drives: two
    # `check_lazy_status` rounds (ComfyUI's lazy protocol, re-invoked as the wired scalar
    # cooks) and then the node's `execute`.
    "lazy_required_bindings":  3,    # 2 lazy rounds + `prepare`'s E6003 forgiveness gate,
                                     # which the node reaches with `forgive_dead_refs=True`
                                     # whenever a slot map exists. THE row this scenario
                                     # exists for, and the only non-zero reading of it in the
                                     # whole file — every other scenario pins it at 0 by
                                     # never reaching the tier at all.
    "Lexer.tokenize":          0,    # PERF-4's class, gated from here on. A slider tick cost
    "Parser.parse":            0,    # TWO of each before PERF-4 (24 lexes and 24 parses per
                                     # 12 ticks: one per lazy round, the E6003 gate sharing
                                     # the second round's memo key) and ONE in total after,
                                     # paid by the first tick, because `tex_lazy` now
                                     # memoizes the parse per SOURCE and folds a
                                     # `clone_tree` copy per value. A non-zero here means a
                                     # scrub re-reads a program that did not change.
    "TEXCache.compile_ast":    0,    # ANIM-1 on the node path: a moving $param is a cook-time
                                     # binding, never a recompile.
    "TEXCache.compile_tex":    1,    # one cached-compile lookup per execute,
    "TEXCache.fingerprint":    1,    # and one key for it (PERF-5's one-per-cook shape).
    "TypeChecker.check":       0,
    "TypeChecker.check_collect": 0,
    "tex_engine.cook":         0,    # 0, NOT a miscount: `tex_node.execute` calls `prepare`
    "tex_engine.prepare":      1,    # and `run` itself, because it needs the plan between
    "tex_engine.run":          1,    # them (`fused_chain` for the Q-4 stage attribution).
                                     # `cook` is the one-call convenience the other scenarios
                                     # use. A 1 in the cook row would mean the node stopped
                                     # being able to see its own plan.
    "tex_roi._fold_program":   0,    # no ROI on this path at all: ComfyUI has no viewport
    "tex_roi.roi_plan":        0,    # window to cook, so the whole ROI/results-cache tier
    "tex_roi.chain_windows":   0,    # the other six scenarios exercise is absent here. That
    "tex_results.lineage_key": 0,    # is the POINT of the scenario — it is the other half of
    "ResultCache.get":         0,    # what an embedding host pays, not a second reading of
    "ResultCache.put":         0,    # the same half.
    "tex_memory.run_roi":      0,
    "Interpreter._exec_stmt":  2,    # the program's two statements, on the CPU interpreter
                                     # tier (see `_TERMINAL`'s note: a compiled tier is a
                                     # tier change and a CHANGELOG entry).
}


_CHECKPOINT_SERVE = {
    # BENCH-4. `docs/host-path-counts.md` §3 named this gap: no scenario drove
    # `tex_checkpoint.cook_checkpointed`, the CACHE-7 checkpoint-serve path a second
    # embedding host's router sends its commonest interactive edit to. This scenario settles
    # a PROF-1 cost table on its own two-stage chain (a heavy triple-blur, then the ONE stage
    # a slider drags), materializes the one checkpoint that chain can ever have, then drives
    # steady ticks that edit the terminal stage and call the checkpointed cook for real —
    # the planner runs every tick, off the now-frozen table, rather than a hand-fed `cuts=`.
    "tex_checkpoint.cook_checkpointed": 1,  # THE row this scenario exists for: one served
                                     # cook per tick. 0 on every other scenario in this file —
                                     # none of them ever build a ResultCache-backed linear
                                     # chain with a settled cost table.
    "tex_engine.boundary_lineage_key": 1,   # one probe. A two-stage chain has exactly one
                                     # possible cut, which is therefore also the DEEPEST, so
                                     # `cook_checkpointed`'s deepest-first loop hits on the
                                     # FIRST probe every tick — never more than one per
                                     # planned cut, and this chain plans exactly one.
    "lazy_required_bindings":  1,   # the host's OWN planner calling the analysis DIRECTLY,
                                     # never through `prepare()` — a second, independent door
                                     # onto this tier besides `node_scrub`'s
                                     # `forgive_dead_refs` one. A 0 here would mean the
                                     # planner stopped asking; a value > 1 would mean it
                                     # asks more than once per tick for no new information.
    "ResultCache.get":          1,  # one boundary read, and it HITS — `materialize()` ran in
                                     # `build()`, outside the counted region, so a steady tick
                                     # is a served checkpoint, never a first-cook fallback.
    "ResultCache.put":          0,  # the checkpoint was materialized ONCE, before any tick
                                     # was counted; a non-zero here means a tick re-harvested
                                     # a boundary it already has.
    "TEXCache.compile_ast":     0,  # ANIM-1: the terminal stage's $knob is a cook-time
                                     # binding, never a recompile.
    "TEXCache.compile_tex":     1,  # one cached-compile lookup for the served suffix (the
                                     # terminal stage alone — the blur is never re-cooked).
    "TEXCache.fingerprint":     1,
    "TypeChecker.check":        0,
    "TypeChecker.check_collect": 0,
    "Lexer.tokenize":           0,  # the suffix's program was already compiled during
    "Parser.parse":             0,  # `build()`'s settling loop; a steady tick re-reads
                                     # nothing.
    "tex_engine.cook":          0,  # `cook_checkpointed` never calls `tex_engine.cook` /
    "tex_engine.prepare":       0,  # `.prepare` / `.run` — it dispatches straight to
    "tex_engine.run":           0,  # `cook_stage_list`'s interpreter, same as `node_scrub`'s
                                     # `execute` does NOT (contrast `_NODE_SCRUB`'s prepare=1/
                                     # run=1): this is a THIRD shape, not a repeat of the
                                     # second.
    "tex_roi._fold_program":    0,  # no ROI tier at all: `cook_checkpointed` takes no window,
    "tex_roi.roi_plan":         0,  # by construction — it always cooks the whole (served)
    "tex_roi.chain_windows":    0,  # frame. The comp's ROI/results-cache tier the first seven
                                     # scenarios exercise is simply absent from this path.
    "tex_memory.run_roi":       0,
    "Interpreter._exec_stmt":   1,  # the terminal stage's one statement — the served suffix
                                     # is that stage ALONE, never the blur (see the class
                                     # docstring's deepest-first argument).
}


_INTERP_CHAIN_SCRUB = {
    # v0.42's ASK-shaped scenario: an embedding host's interactive floor. Every stage is
    # uncached (`use_cache=False`), the window pans AND the terminal `$param` moves on the
    # SAME tick — no memo above ever sees both misses together. This is the scenario
    # TRK-64/TRK-65/TRK-72/TRK-84 each prove themselves against (see the CUDA table below
    # for TRK-84's kernel/alloc delta).
    "tex_engine.cook":          10,   # every stage is dirty AND uncached: all ten cook.
    "TEXCache.compile_ast":      0,   # ANIM-1 across the whole chain.
    "TEXCache.compile_tex":     10,
    "TEXCache.fingerprint":     10,   # PERF-5 shape: 1 per cook, ten cooks.
    "Lexer.tokenize":            0,   # PERF-1's parse memo: ten sources, all already parsed.
    "Parser.parse":               0,
    "tex_roi._fold_program":      1,  # PERF-1's memo, unmoved: nine stages' $param values are
                                      # unchanged tick to tick, so their walk memo hits even
                                      # though the WINDOW moved (`_walk`'s key carries param
                                      # values, not the window); only the terminal stage's
                                      # moved knob misses and re-folds. TRK-65's fix does not
                                      # move THIS row — it makes that one miss's own
                                      # region-dependence recheck a cached lookup
                                      # (`_unfolded_region_independent`) instead of a fresh
                                      # `_ControlFlowLint` fixed-point walk, visible in the
                                      # frame counts, not this API row.
    "tex_roi.roi_plan":          11,  # 10 engine plans (one per stage) + 1 host halo question.
    "tex_roi.chain_windows":      1,  # one plan per tick, whatever the window.
    "ResultCache.get":            0,  # `use_cache=False`: the cache tier is never probed —
    "ResultCache.put":            0,  # the honest reading of "every node between the nearest
    "results_cache.entries_added": 0, # cache and the viewer is uncached" (an embedding host's
                                      # own caching policy, never TEX's to enforce).
    "tex_memory.run_roi":         10, # every stage stayed on the ROI path.
    "enforce_cache_budget":       10, # TRK-72: counted here, once per cook — ten times a
                                      # tick, the largest per-tick multiplier any scenario in
                                      # this file gives it. No cheaper implementation exists
                                      # without changing eviction behaviour (see the tracker
                                      # row); this is the count a future fix would move.
    "_disown_inputs":            10,
    "_tile_plan":                  0,  # 96^2/48^2 gate shape never crosses the tile threshold.
    "_halo_tile_plan":             0,
    "host.get_free_memory":       0,  # off CUDA the planners return before the query.
    "lazy_required_bindings":     0,  # this path never reaches the lazy tier at all (RoiComp
                                      # drives `tex_engine.cook` directly, like every other
                                      # comp scenario — see `_NODE_SCRUB`'s own note on why
                                      # that tier needs `forgive_dead_refs`).
    "tex_checkpoint.cook_checkpointed": 0,
    "tex_engine.boundary_lineage_key":   0,
    "Interpreter._exec_stmt":    12,  # 12 statements across the ten stage programs — the
                                      # same reading `_ALL_DIRTY` pins, for the same reason
                                      # (all ten stages cook).
}


def test_bench2_interactive_per_tick_counts(r: SubTestResult):
    """The gate: the device-independent per-tick counts of the seven interactive paths."""
    print("\n--- BENCH-2: per-tick structural counts (CPU, PROF-1 disarmed) ---")
    for label, pins in (("terminal", _TERMINAL), ("midgraph", _MIDGRAPH), ("pan", _PAN),
                        ("all_dirty", _ALL_DIRTY), ("lint", _LINT),
                        ("node_scrub", _NODE_SCRUB),
                        ("checkpoint_serve", _CHECKPOINT_SERVE),
                        ("interp_chain_scrub", _INTERP_CHAIN_SCRUB)):
        try:
            _check(r, label, _api_counts(label), pins)
        except Exception as e:
            r.fail(f"BENCH-2 {label}", f"{type(e).__name__}: {e}")


def test_bench4_checkpoint_serve_settles_in_147_cooks(r: SubTestResult):
    """BENCH-4's precondition, pinned rather than assumed.

    `plan_checkpoints` returns `[]` (cook exactly as today) until PROF-1 has
    `tex_checkpoint.MIN_SAMPLES` (12) samples of a key, and the sampling rule is every cook of
    an UNSEEN key for the first `_WARMUP_SAMPLES` (3), then one in `_SAMPLE_EVERY` (16) after
    that — `tex_runtime/profile.py`'s own private schedule. Reaching 12 costs
    `3 + (12 - 3) * 16` = **147** cooks, and this is that arithmetic's canary: it moves the
    moment any of the three private constants above move, which is the point — a sampling-
    schedule change is real news for a host whose router waits on this table (see
    `docs/host-path-counts.md` §6 item 4's PROF-1 note), not a silent drift this file should
    absorb. `CheckpointServeScenario.build()` measures the real count rather than asserting it
    (`self.cooks_to_settle`); this test reads the same number back."""
    print("\n--- BENCH-4: the checkpoint-serve cost table settles in 147 cooks ---")
    b = _bench()
    cls = next(c for c in b.SCENARIOS if c.name == "checkpoint_serve")
    scn = cls(RES, WINDOW, "cpu", ticks=TICKS)
    try:
        b.pass_api(scn, TICKS)
        warm, cuts = scn.cooks_to_settle, scn.cuts_planned
    finally:
        scn.teardown()
    if warm != 147:
        r.fail("BENCH-4 settling cost", f"cooks_to_settle: 147 -> {warm} — PROF-1's warmup/"
               f"sample-every/MIN_SAMPLES schedule moved; re-derive the arithmetic in this "
               f"test's docstring and explain the move in CHANGELOG.md")
    elif cuts != [1]:
        r.fail("BENCH-4 cut plan", f"cuts_planned: [1] -> {cuts} — the two-stage chain's "
               f"margin over the materialization floor (see the scenario's class docstring) "
               f"did not hold on this box")
    else:
        r.ok(f"checkpoint_serve: settled in {warm} cooks, cuts={cuts}")


def test_bench2_no_engine_side_cuda_sync_on_an_interactive_tick(r: SubTestResult):
    """With PROF-1 DISARMED, TEX itself issues no `torch.cuda.synchronize` on an interactive
    tick. A sync the host did not ask for is a pipeline stall charged to somebody else's
    frame, and PROF-1's own four-per-sampled-cook syncs are the reason the profiler is
    disarmed by default (`tex_runtime/profile.py`, invariant #7).

    Counted with the CALLER's file, so the demo host's own end-of-frame barrier
    (`RoiComp.cook`, which is host policy) is a different row and does not mask this one.
    The row is zero on CPU too — nothing calls it — so the assertion is portable; the CUDA
    reading that gives it teeth is in the CUDA test below."""
    print("\n--- BENCH-2: zero engine-side CUDA syncs per interactive tick ---")
    for label in ("terminal", "midgraph", "pan", "all_dirty", "node_scrub",
                 "checkpoint_serve", "interp_chain_scrub"):
        try:
            got = _api_counts(label)
            lo, hi = got.get("torch.cuda.synchronize[engine]", (None, None))
            if lo is None:
                r.fail("BENCH-2 sync row", "the harness stopped reporting "
                       "torch.cuda.synchronize[engine]")
            elif (lo, hi) != (0, 0):
                r.fail("BENCH-2 sync", f"{label}: TEX issued {lo}..{hi} torch.cuda.synchronize "
                       f"call(s) per tick with PROF-1 disarmed (expected 0) — re-derive with "
                       f"`{REDERIVE}` and explain in CHANGELOG.md")
            else:
                r.ok(f"{label}: 0 engine-side torch.cuda.synchronize per tick")
        except Exception as e:
            r.fail(f"BENCH-2 sync {label}", f"{type(e).__name__}: {e}")


# ── CUDA-only: the device rows ──────────────────────────────────────────────
# Kernel launches, 4-byte `.item()` drains and allocator allocations per tick, measured at
# 1024^2 with a 512^2 window on an sm_120 device. These are DEVICE-DEPENDENT by nature (a
# different fuser or a different tier emits a different number of kernels), so they SKIP
# rather than pass without CUDA — a skipped row is visible, a passed one is a lie.
_CUDA_RES, _CUDA_WINDOW, _CUDA_TICKS = 1024, 512, 4
_CUDA_PINS = {
    #                    kernels   D2H memcpys   allocations
    "terminal":         (22,       0,            18),
    "midgraph":         (74,       0,            56),   # PERF-2 re-pin (was 2 — the two
    "pan":              (26,       0,            22),   #   `gauss_blur` stages reading their
    "all_dirty":        (112,      0,            86),   #   sigma back; was 3 with `glow`).
    # BENCH-4: measured identically at this shape (1024^2/512^2) and at the gate shape
    # (96^2/48^2) — the served suffix is the terminal stage alone, resolution-independent in
    # KIND if not in exact device work, and both shapes agreed on this box.
    "checkpoint_serve": (6,        0,            4),
    # v0.42 ASK: ten uncached stages cook every tick (the `all_dirty`-shaped ten cooks) AND
    # the window pans every tick (the `pan`-shaped LAT-4 coordinate-builtin miss) on the SAME
    # tick — the shape TRK-84's fix (this file's own follow-up commit) is measured against.
    "interp_chain_scrub": (125,    0,            88),
}
# WHY THE D2H COLUMN IS NOW ZERO EVERYWHERE, AND WHAT WOULD MAKE IT NON-ZERO AGAIN.
# `gauss_blur` needs a Python number for its kernel radius and used to get it with
# `sigma_t.item()` — a 4-byte device->host copy plus the stream synchronisation it implies,
# once per blur stage per cook. PERF-2 carries the host reading of a literal / `$param` /
# folded constant ON the 0-dim tensor it is minted into, so the number is resolved on the
# host. A sigma genuinely COMPUTED on the device still reads back, and correctly so — the
# comp these scenarios drive has no such stage, which is why this column reads 0 rather than
# "fewer". A future stage whose sigma is an expression over a binding would legitimately put
# it back; that is a comp change, not a regression, and belongs in this comment when it
# happens.
_CUDA_REDERIVE = (f"python benchmarks/host_path_counts.py --device cuda --res {_CUDA_RES} "
                  f"--window {_CUDA_WINDOW} --ticks {_CUDA_TICKS} --prof1 off")


def test_bench2_cuda_per_tick_counts(r: SubTestResult):
    print("\n--- BENCH-2: per-tick CUDA kernels / D2H drains / allocations ---")
    if not torch.cuda.is_available():
        r.skip("BENCH-2 CUDA counts", "no CUDA device — device rows are not measurable here")
        return
    b = _bench()
    for label, (kern, d2h, allocs) in _CUDA_PINS.items():
        try:
            cls = next(c for c in b.SCENARIOS if c.name == label)
            scn = cls(_CUDA_RES, _CUDA_WINDOW, "cuda", ticks=_CUDA_TICKS)
            try:
                scn.epoch = 0
                api = b.pass_api(scn, _CUDA_TICKS)
                scn.epoch = 2
                cuda = b.pass_cuda(scn, _CUDA_TICKS)
            finally:
                scn.teardown()
            got = {"cuda.kernels": cuda.get("cuda.kernels", {}),
                   "cuda.memcpy_DtoH": cuda.get("cuda.memcpy_DtoH", {}),
                   "alloc.allocated": api.get("alloc.allocated", {})}
            want = {"cuda.kernels": kern, "cuda.memcpy_DtoH": d2h, "alloc.allocated": allocs}
            bad = []
            for row, exp in want.items():
                s = got[row]
                if not s:
                    bad.append(f"{row}: MISSING")
                elif not s["stable"]:
                    bad.append(f"{row}: unstable {s['min']}..{s['max']} (not gateable here)")
                elif s["min"] != exp:
                    bad.append(f"{row}: {exp} -> {s['min']} per tick")
            if bad:
                r.fail(f"BENCH-2 cuda {label}", "; ".join(bad) +
                       f" || re-derive with `{_CUDA_REDERIVE}`; a kernel-count move on the same "
                       f"source is a fuser/tier change and belongs in the CHANGELOG")
            else:
                r.ok(f"{label}: {kern} kernels / {d2h} D2H / {allocs} allocations per tick")
        except Exception as e:
            r.fail(f"BENCH-2 cuda {label}", f"{type(e).__name__}: {e}")


# ── PERF-6: how often a tick asks the host how much VRAM is free ────────────
# `host.get_free_memory` is the seam that COSTS the money, and it is NOT the same
# measurement as the `torch.cuda.mem_get_info` row beside it: the driver call is the inner
# ~13-17 us of a ~90-112 us host call (the host folds allocator statistics in on top), so a
# fix measured on `mem_get_info` alone would claim a seventh of what it saved.
#
# The counts are RESOLUTION-INDEPENDENT (verified: identical at 96^2 and at 1024^2 on an
# sm_120 box), because the query sits before the budget arithmetic that resolution moves —
# so this test runs at the cheap CPU shape with the device flipped, rather than paying for a
# second 1024^2 matrix.
_FREE_MEM_CPU = 0       # every scenario: the planners return before the query off CUDA
_FREE_MEM_CUDA = {
    # PERF-6 re-pin (was 7). A whole-frame recook is ten cooks, of which the seven POINTWISE
    # stages reach `_tile_plan`'s free-VRAM question (the three blur/morphology stages are not
    # tile-safe, so they leave through `is_tile_safe_cached` and `_halo_tile_plan`'s cheap
    # gate). All seven are now answered from the last live reading, because a cook this far
    # under the budget cannot be the cook that needs the number: `tex_tiling._free_foreign`
    # holds only the bytes torch's allocator does not own, and re-reads `torch_allocated` on
    # every call. A NON-ZERO here means a stage got close enough to the budget for the margin
    # to fail — which is the design working, not a regression — or that the decomposition
    # stopped resolving (no `device_total_mem`, no allocator statistics).
    "all_dirty":   0,
    # PERF-6 re-pin (was 4). The first whole frame after a source edit: seven cooks, four of
    # them pointwise, same reasoning.
    "source_edit": 0,
    # The interactive ticks pay NOTHING: `_preflight_memory` fires (1 / 5 / 1 per tick) and
    # LAT-2's cheap path returns before the query, and the ROI route never reaches a tile plan.
    "terminal":    0,
    "midgraph":    0,
    "pan":         0,
    # `tex_api.check` never cooks, so nothing asks.
    "lint":        0,
    # NOT the tile planner's, and the reason this row cannot be read as "the planner's
    # queries": a prewarm asks `_cuda_headroom_ok` (tex_runtime/compiled.py) once per program
    # before submitting a background compile. It is also this row's NON-INERT witness — a pin
    # of 0 on five of the seven scenarios above would otherwise be satisfied by a dead spy.
    "prewarm":    10,
}


class _FakeModelManagement:
    """The whole of `comfy.model_management` that `ComfyHostServices` needs to answer
    "how much is free?" — a fake host the harness drives, so the CUDA-only implementation
    is witnessed on a CPU-only box without a device and without importing ComfyUI."""
    def get_free_memory(self, device):
        return 1234.0

    def processing_interrupted(self):
        return False


def _witness_free_memory_spy(r: SubTestResult) -> bool:
    """CPU NON-INERT WITNESS for the five `_FREE_MEM_CPU` zeros below.

    `_FREE_MEM_CUDA["prewarm"] = 10` is this row's only non-zero reading, and it exists on
    the CUDA leg — which CI, being CPU-only, never runs. So on CI the pins below asserted
    `0 == 0` five times with nothing showing that the `host.get_free_memory` spy had been
    installed at all: a renamed target, a moved class or a module imported under a second
    name would have satisfied every one of them. That is the ANIM-1 failure this file's
    mutation guard exists to prevent, surviving in the one row the guard does not cover
    (its cold prewarm tick reads 0 here for the same device reason).

    The witness drives the SEAM rather than a scenario, because off CUDA the planners
    legitimately return before ever asking — the honest thing to prove on CPU is not "a tick
    queries" but "if a tick queried, this row would count it". BOTH patched implementations
    are driven, since which one a run uses depends on whether ComfyUI is importable and
    patching only one reports a confident zero in the other shape.
    """
    b = _bench()
    from TEX_Wrangle.tex_runtime import host as _host
    row = "host.get_free_memory"
    try:
        with b.CallSpies({row: b.SPY_TARGETS[row]}, sync_caller_attribution=False) as spies:
            dev = torch.device("cpu")
            _host.NullHostServices().get_free_memory(dev)
            _host.ComfyHostServices(_FakeModelManagement()).get_free_memory(dev)
            n = spies.snapshot().get(row, 0)
    except Exception as e:
        r.fail("BENCH-2 free-memory witness (cpu)", f"{type(e).__name__}: {e}")
        return False
    if n == len(b.SPY_TARGETS[row]):
        r.ok(f"cpu witness: {row} counted {n} direct calls (one per patched implementation) "
             f"— the zeros below are measured, not vacuous")
        return True
    r.fail("BENCH-2 inert free-memory spy",
           f"{row} counted {n} of {len(b.SPY_TARGETS[row])} direct calls through the very "
           f"attributes the harness patches — the spy is not installed, so every 0 pinned "
           f"for this row is vacuous")
    return False


def test_bench2_free_memory_queries_per_tick(r: SubTestResult):
    """PERF-6: the number of live host free-VRAM queries an interactive tick pays."""
    print("\n--- BENCH-2: host free-VRAM queries per tick ---")
    _witness_free_memory_spy(r)
    for label in ("terminal", "midgraph", "pan", "all_dirty", "lint"):
        try:
            _check(r, f"{label} (cpu)", _api_counts(label),
                   {"host.get_free_memory": _FREE_MEM_CPU})
        except Exception as e:
            r.fail(f"BENCH-2 free-memory {label} (cpu)", f"{type(e).__name__}: {e}")

    if not torch.cuda.is_available():
        r.skip("BENCH-2 free-memory (cuda)", "no CUDA device — the planners return off CUDA, "
               "so the only readings that can move are not measurable here")
        return
    for label, want in _FREE_MEM_CUDA.items():
        try:
            _check(r, f"{label} (cuda)", _api_counts(label, "cuda"),
                   {"host.get_free_memory": want})
        except Exception as e:
            r.fail(f"BENCH-2 free-memory {label} (cuda)", f"{type(e).__name__}: {e}")


#: An attribute name no code object carries on any interpreter. Setting the harness's
#: `_QUALNAME_ATTR` to it makes `frame_qualname` take the branch Python 3.10 takes.
_FORCED_QUALNAME_ATTR = "co_qualname_absent_before_python_3_11"


def _frame_rows(b, forced: bool, tag: str) -> dict:
    """One real front-end call under the harness's OWN `FrameCounter`, so the row names the
    `frames.*` family is built from are the thing measured — not a copy of its keying."""
    from TEX_Wrangle.tex_cache import parse_and_split
    src = "@OUT = vec4(@A.rgb * 1.25, 1.0);\n// " + tag + "\n"
    saved = b._QUALNAME_ATTR
    if forced:
        b._QUALNAME_ATTR = _FORCED_QUALNAME_ATTR
    try:
        fc = b.FrameCounter()
        with fc:
            parse_and_split(src, {})
        return dict(fc.counts)
    finally:
        b._QUALNAME_ATTR = saved


def test_bench2_frame_rows_survive_a_missing_co_qualname(r: SubTestResult):
    """PY-3.10 — a `frames.*` row name may not depend on `co_qualname`, which is 3.11+.

    CI runs 3.10, 3.11 and 3.12. Below 3.11 a code object carries only `co_name`, so a method
    frame would key `tex_compiler.lexer:tokenize` where every reader of this census — and
    every pin spelled against it in `tests/test_perf7_compiled_cold.py` — reads
    `tex_compiler.lexer:Lexer.tokenize`. A row that is spelled two ways depending on the
    interpreter is a row that counts zero on one of them, which is the silent-zero failure
    the filter probe above already exists for, one layer down.

    Neither development box has a 3.10, so the fallback is forced through the harness's one
    seam and required to produce the SAME key the native path produces here. Real proof is
    the Linux 3.10 leg of CI.

    ON a 3.10 interpreter both legs take the fallback and the comparison degenerates into the
    measurement itself — the rows are still required to be counted under their qualified
    names, which is the thing that was red there. So this row is not guarded by an interpreter
    check and does not skip."""
    print("\n--- BENCH-2: frame row names survive a missing co_qualname (Python 3.10) ---")
    b = _bench()
    try:
        native = _frame_rows(b, False, "native")
        forced = _frame_rows(b, True, "forced")
    except Exception as e:
        r.fail("BENCH-2 qualname fallback", f"{type(e).__name__}: {e}")
        return
    if b._QUALNAME_ATTR != "co_qualname":
        r.fail("BENCH-2 qualname seam", f"the seam was left at {b._QUALNAME_ATTR!r}: every "
               f"later row in this process would be measured through the fallback")
        return
    for row, bare in (("tex_compiler.lexer:Lexer.tokenize", "tex_compiler.lexer:tokenize"),
                      ("tex_compiler.parser:Parser.parse", "tex_compiler.parser:parse")):
        n, fb = native.get(row, 0), forced.get(row, 0)
        if n < 1:
            r.fail("BENCH-2 qualname drive", f"{row} counted 0 under the NATIVE resolver — "
                   f"the drive does not reach it, so this row proves nothing about 3.10")
        elif fb < 1:
            r.fail("BENCH-2 qualname fallback", f"{row} counted {fb} with `co_qualname` "
                   f"forced missing ({bare!r} read {forced.get(bare, 0)} instead): on Python "
                   f"3.10 this census row is spelled differently from every pin against it")
        elif forced.get(bare, 0):
            r.fail("BENCH-2 qualname fallback", f"the fallback also wrote the bare-name row "
                   f"{bare!r} ({forced[bare]}): two spellings of one function is the drift "
                   f"this row exists to catch")
        else:
            r.ok(f"{row}: {n} native / {fb} with the pre-3.11 fallback forced, same key")


def test_bench2_counters_are_not_inert(r: SubTestResult):
    """MUTATION GUARD — the ANIM-1 lesson applied to this file.

    Most of the pins above are of the form "this row is exactly N", and several of the most
    load-bearing N are ZERO (`compile_ast` on every scrub, the whole of `lint`). A spy list
    that silently failed to install — a renamed target, a `@staticmethod` patched as a plain
    function, a module imported under a second name — would satisfy every one of them, and
    the suite would report a contract it was no longer measuring.

    So a COLD scenario is driven and the same counters are required to be NON-ZERO, and the
    frame counter is required to see TEX frames at all. A harness that cannot fail does not
    protect the contract it is pointed at."""
    print("\n--- BENCH-2 mutation guard: the counters fire on a cold tick ---")
    b = _bench()

    # The frame counter's FILTER, before any scenario runs. `frames.*` is a whole family of
    # rows, and the filter decides which frames are looked at — so a filter that accepts one
    # spelling of this package's path and not another zeroes the entire family at once,
    # without failing. That is not hypothetical: `Path.resolve()` follows the
    # `custom_nodes\TEX_Wrangle -> TEX` junction while imported modules keep the junction
    # spelling in `co_filename`, and a hook comparing one against the other matched nothing.
    # Drive a synthetic frame under EACH accepted spelling, and one from outside.
    class _Code:
        co_firstlineno, co_name, co_qualname = 1, "probe", "probe"

        def __init__(self, fn):
            self.co_filename = fn

    class _Frame:
        def __init__(self, fn):
            self.f_code = _Code(fn)

    bad = []
    for pref in b._PKG_PREFIXES:
        fc = b.FrameCounter()
        fc._hook(_Frame(pref + "tex_engine.py"), "call", None)
        if fc.counts.get("tex_engine:probe") != 1:
            bad.append(f"a frame spelled {pref!r} was NOT counted ({dict(fc.counts)})")
    for fn in (os.path.join(tempfile.gettempdir(), "not_tex.py"),
               os.path.join(b._BENCH_PREFIXES[0], "host_path_counts.py")):
        fc = b.FrameCounter()
        fc._hook(_Frame(fn), "call", None)
        if fc.counts:
            bad.append(f"{fn!r} must not be counted, got {dict(fc.counts)}")
    r.fail("BENCH-2 frame filter", "; ".join(bad)) if bad else \
        r.ok(f"the frame filter accepts all {len(b._PKG_PREFIXES)} spelling(s) of the package "
             f"and neither the harness's own frames nor anything outside it")

    scn = b.PrewarmScenario(RES, WINDOW, "cpu", ticks=1)
    try:
        cold = b.pass_api(scn, 1)
    finally:
        scn.teardown()
    for row in ("TEXCache.compile_ast", "TEXCache.compile_tex", "Lexer.tokenize",
                "Parser.parse", "TEXCache.fingerprint", "TypeChecker.check"):
        n = cold.get(row, {}).get("min", 0)
        r.ok(f"cold prewarm tick: {row} = {n} (> 0, so the spy is live)") if n > 0 else \
            r.fail("BENCH-2 inert spy", f"{row} counted 0 on a COLD tick — the spy is not "
                   f"installed, so every zero-valued pin in this file is vacuous")

    scn = b.TerminalKnobScenario(RES, WINDOW, "cpu", ticks=1)
    scn.epoch = 9
    try:
        frames = b.pass_frames(scn, 1, 3)
    finally:
        scn.teardown()
    tot = frames.get("frames.total", {}).get("min", 0)
    r.ok(f"terminal tick: {tot} TEX python frames counted (> 0, so the profiler hook is live)") \
        if tot > 0 else \
        r.fail("BENCH-2 inert frames", "the sys.setprofile frame counter saw no TEX frames")

    # And the negative half of the mutation guard: a cook-shaped scenario must NOT look like
    # the lint one. If these two ever agreed, one of them is not running what it says.
    term = _api_counts("terminal")
    lint = _api_counts("lint")
    r.ok("terminal and lint disagree on tex_engine.cook (the scenarios are distinct)") \
        if term.get("tex_engine.cook") != lint.get("tex_engine.cook") else \
        r.fail("BENCH-2 scenarios", "terminal and lint report the same cook count — one of "
               "the two scenarios is not driving what its name says")
